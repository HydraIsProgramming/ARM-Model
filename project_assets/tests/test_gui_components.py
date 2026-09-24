"""Tests for GUI components (configuration, kinematics, control)"""

import sys
from pathlib import Path
import pytest
import numpy as np
import json
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_armMotion.config import ArmConfiguration
from rl_armMotion.utils import (
    ArmKinematics,
    ArmController,
    MotionRecorder,
    ArmState,
)


class TestArmConfiguration:
    """Test arm configuration system"""

    def test_default_configuration(self):
        """Test default configuration initialization"""
        config = ArmConfiguration()
        assert config.dof == 2
        assert len(config.link_lengths) == 2
        assert len(config.masses) == 2
        assert len(config.inertias) == 2
        assert len(config.initial_angles) == 2

    def test_preset_loading(self):
        """Test loading presets"""
        presets = ArmConfiguration.list_presets()
        assert "2dof_simple" in presets
        assert "simple_planar" in presets

        config = ArmConfiguration.get_preset("2dof_simple")
        assert config.name == "2DOF_Simple_Arm"
        assert config.dof == 2

        config_planar = ArmConfiguration.get_preset("simple_planar")
        assert config_planar.dof == 3

    def test_json_serialization(self):
        """Test JSON save/load round trip"""
        config = ArmConfiguration.get_preset("7dof_industrial")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            config.to_json(filepath)
            assert Path(filepath).exists()

            loaded_config = ArmConfiguration.from_json(filepath)
            assert loaded_config.dof == config.dof
            assert loaded_config.name == config.name
            assert np.allclose(loaded_config.link_lengths, config.link_lengths)
        finally:
            Path(filepath).unlink()

    def test_validation(self):
        """Test configuration validation"""
        config = ArmConfiguration()
        is_valid, msg = config.validate()
        assert is_valid is True

        # Create invalid config
        bad_config = ArmConfiguration(masses=[-1.0] * 2)
        is_valid, msg = bad_config.validate()
        assert is_valid is False

    def test_get_joint_limits(self):
        """Test joint limits access"""
        config = ArmConfiguration()
        limits = config.get_joint_limits()
        assert limits.shape == (2, 2)
        assert np.all(limits[:, 0] < limits[:, 1])  # Min < Max


class TestArmKinematics:
    """Test forward kinematics"""

    def test_forward_kinematics_zero_angles(self):
        """Test FK with initial downward configuration"""
        config = ArmConfiguration()
        angles = np.array([-np.pi/2, 0])  # Vertical downward
        positions = ArmKinematics.forward_kinematics(angles, config)

        # Should have 3 positions (2 links + base)
        assert positions.shape == (3, 3)

        # Base should be at origin
        np.testing.assert_array_almost_equal(positions[0], [0, 0, 0])

        # End-effector should be along negative y-axis (pointing down)
        assert positions[-1, 0] == pytest.approx(0, abs=1e-6)  # x ≈ 0
        assert positions[-1, 1] < 0  # y < 0 (down)

    def test_forward_kinematics_simple(self):
        """Test FK with simple planar arm"""
        config = ArmConfiguration.get_preset("simple_planar")
        angles = np.array([0, 0, 0])
        positions = ArmKinematics.forward_kinematics(angles, config)

        # Should have 4 positions (3 joints + base)
        assert positions.shape == (4, 3)

        # Check roughly correct positioning
        total_length = sum(config.link_lengths[:3])
        assert positions[-1, 0] == pytest.approx(total_length, abs=0.01)

    def test_end_effector_position(self):
        """Test end-effector position computation"""
        config = ArmConfiguration()
        angles = np.zeros(2)
        ee_pos = ArmKinematics.end_effector_position(angles, config)

        assert len(ee_pos) == 3
        assert ee_pos[0] > 0  # Should be along x-axis


class TestArmController:
    """Test arm controller"""

    def test_initialization(self):
        """Test controller initialization"""
        config = ArmConfiguration()
        controller = ArmController(config)

        assert controller.dof == 2
        assert len(controller.angles) == 2
        np.testing.assert_array_almost_equal(controller.angles, [-np.pi/2, 0])

    def test_update_joint_angle(self):
        """Test updating joint angle"""
        config = ArmConfiguration()
        controller = ArmController(config)

        # Update joint 0
        target = 1.0
        controller.update_joint_angle(0, target)
        assert controller.angles[0] == pytest.approx(target)

        # Update with out-of-bounds value (should clamp)
        out_of_bounds = 10.0
        controller.update_joint_angle(0, out_of_bounds)
        assert controller.angles[0] <= config.joint_limits_max[0]

    def test_increment_joint(self):
        """Test incremental joint motion"""
        config = ArmConfiguration()
        controller = ArmController(config)

        initial_angle = controller.angles[0]
        controller.increment_joint(0, 0.1)
        assert controller.angles[0] > initial_angle

    def test_home_position(self):
        """Test moving to home position"""
        config = ArmConfiguration()
        controller = ArmController(config)
        home_angles = np.array([-np.pi/2, 0])

        # Move away from home
        controller.update_joint_angle(0, 1.0)
        assert controller.angles[0] != home_angles[0]

        # Return to home
        controller.set_home_position()
        np.testing.assert_array_almost_equal(controller.angles, home_angles)

    def test_get_state(self):
        """Test state retrieval"""
        config = ArmConfiguration()
        controller = ArmController(config)

        state = controller.get_state(timestamp=1.0)
        assert isinstance(state, ArmState)
        assert state.timestamp == 1.0
        assert len(state.angles) == 2


class TestMotionRecorder:
    """Test motion recording and playback"""

    def test_initialization(self):
        """Test recorder initialization"""
        recorder = MotionRecorder()
        assert recorder.get_num_frames() == 0
        assert not recorder.is_recording

    def test_recording(self):
        """Test recording frames"""
        config = ArmConfiguration()
        controller = ArmController(config)
        recorder = MotionRecorder()

        recorder.start_recording()
        assert recorder.is_recording

        # Record some frames
        for i in range(10):
            controller.increment_joint(0, 0.05)
            state = controller.get_state()
            recorder.record_frame(state)

        recorder.stop_recording()
        assert not recorder.is_recording
        assert recorder.get_num_frames() == 10

    def test_json_save_load(self):
        """Test motion JSON serialization"""
        config = ArmConfiguration()
        controller = ArmController(config)
        recorder = MotionRecorder()

        recorder.start_recording()
        for i in range(5):
            controller.increment_joint(0, 0.05)
            state = controller.get_state()
            recorder.record_frame(state)
        recorder.stop_recording()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            recorder.save_to_json(filepath)
            assert Path(filepath).exists()

            loaded_recorder = MotionRecorder.load_from_json(filepath)
            assert loaded_recorder.get_num_frames() == 5

            # Compare first frame
            original_frame = recorder.get_frames()[0]
            loaded_frame = loaded_recorder.get_frames()[0]
            np.testing.assert_array_almost_equal(
                original_frame.angles, loaded_frame.angles
            )
        finally:
            Path(filepath).unlink()

    def test_playback(self):
        """Test motion playback"""
        config = ArmConfiguration()
        controller = ArmController(config)
        recorder = MotionRecorder()

        # Record motion
        recorder.start_recording()
        for i in range(5):
            controller.increment_joint(0, 0.05)
            state = controller.get_state()
            recorder.record_frame(state)
        recorder.stop_recording()

        # Playback
        frames = recorder.playback(speed_multiplier=1.0)
        assert len(frames) == 5
        assert all(isinstance(f, ArmState) for f in frames)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_click_snaps_to_the_nearest_holdable_point():
    """A click the arm cannot hold is moved, not discarded.

    Only ~22% of uniformly random clicks land somewhere the arm can hold, so
    refusing the rest means most clicks silently do nothing. The limits are
    real - the policy reaches 0% of targets beyond 0.52 m - so the click is
    honoured as closely as the arm allows instead of being dropped.
    """
    import numpy as np
    from rl_armMotion.two_d.gui.app import ArmControllerGUI
    from rl_armMotion.two_d.evaluation import make_env

    gui = ArmControllerGUI.__new__(ArmControllerGUI)      # the method needs no Tk
    env = make_env(actuation_mode="muscle", waypoints=None)
    shoulder = np.asarray(env.shoulder_base_position, dtype=float)

    # A point well outside the holdable region: straight out at almost full reach.
    far = [float(shoulder[0] + 0.98 * env.reach), float(shoulder[1])]
    assert not env.waypoint_feasibility(far, env.max_curriculum_tolerance).get("feasible", False)
    snapped, moved = gui._nearest_placeable(env, far, [])
    assert snapped is not None
    assert env.waypoint_feasibility(snapped, env.max_curriculum_tolerance)["feasible"]
    assert moved > 0.0
    # It moves toward the arm, not to some arbitrary corner of the workspace.
    assert np.linalg.norm(np.asarray(snapped) - shoulder) < np.linalg.norm(np.asarray(far) - shoulder)

    # A point that is already fine is returned unchanged.
    good = [float(v) for v in env.feasible_points()[0]]
    same, moved0 = gui._nearest_placeable(env, good, [])
    assert moved0 == pytest.approx(0.0, abs=1e-9) and same == pytest.approx(good)


def test_snapping_respects_the_spacing_rule():
    """Snapping must not park two waypoints on top of each other."""
    import numpy as np
    from rl_armMotion.two_d.gui.app import ArmControllerGUI
    from rl_armMotion.two_d.evaluation import make_env

    gui = ArmControllerGUI.__new__(ArmControllerGUI)
    env = make_env(actuation_mode="muscle", waypoints=None)
    first = [float(v) for v in env.feasible_points()[0]]
    # Ask for a second point right on top of the first.
    snapped, _ = gui._nearest_placeable(env, first, [first])
    assert snapped is not None
    assert np.linalg.norm(np.asarray(snapped) - np.asarray(first)) > 2 * env.max_curriculum_tolerance
