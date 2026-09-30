"""The interactive window itself: every section is built, reachable, and shows
the simulator's numbers correctly.

These build the real Tk window (no mainloop) and drive it through its own
handlers. They skip when Tk cannot start - e.g. another GUI already owns this
Python's Tcl interpreter, or there is no display.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "project_assets" / "outputs" / "CHAMPION_HUMAN_V1_R3" / "sac_model_best.zip"


@pytest.fixture(scope="module")
def gui():
    from tkinter import messagebox
    from rl_armMotion.two_d.gui.app import ArmControllerGUI
    try:
        g = ArmControllerGUI()
    except Exception as exc:                      # tkinter.TclError without a usable display
        pytest.skip(f"Tk unavailable: {exc}")
    # A dialog would block the test; fail loudly instead.
    saved = {n: getattr(messagebox, n) for n in ("showinfo", "showwarning", "showerror")}
    for name in saved:
        setattr(messagebox, name, lambda title, msg, _n=name: pytest.fail(f"{_n}: {title}: {msg}"))
    g.create_window()
    g.root.update()
    yield g
    for name, fn in saved.items():
        setattr(messagebox, name, fn)
    g.simulation_active = False
    g.root.destroy()


def test_every_section_and_readout_is_built(gui):
    # 1 model, 2 targets, 3 manual control, 4 arm properties
    for widget in (gui._package_box, gui.sim_toggle_button, gui._pause_button, gui._next_button,
                   gui._place_button, gui.record_button, gui.playback_button):
        assert widget.winfo_exists()
    for key in ("link_len_0", "link_len_1", "mass_0", "mass_1", "damping"):
        assert key in gui.slider_vars
    assert set(gui.angle_labels) == {0, 1}
    # readouts: live status, muscles, all readings, model details, three plots
    assert set(gui._tiles) >= {"state", "episode", "time", "target", "reached", "hold", "distance", "speed",
                               "reward", "holds"}
    assert len(gui._muscle_rows) == 4
    assert gui.metrics_text is not None and gui.sim_model_details_text is not None
    assert gui.torque_ax is not None and gui.velacc_ax_vel is not None and gui.velacc_ax_acc is not None


def test_minimum_size_keeps_the_progress_strip_on_screen(gui):
    gui.root.geometry("1240x740")
    gui.root.update()
    top = gui.root.winfo_rooty() + gui.root.winfo_height()
    for widget in (gui.sim_hold_bar, gui._state_badge, gui._target_label):
        assert widget.winfo_ismapped()
        assert widget.winfo_rooty() + widget.winfo_height() <= top
    assert gui.sim_hold_bar.winfo_width() > 50
    gui.root.geometry("1600x960")
    gui.root.update()


@pytest.mark.skipif(not R3.exists(), reason="R3 package not on disk")
def test_a_run_reports_the_simulators_own_numbers(gui):
    gui._load_model(str(R3))
    gui._wp_choice_var.set("3 waypoints")
    gui._on_waypoint_choice()
    gui._start_simulation()
    assert gui.simulation_active
    n = 60
    for _ in range(n):
        gui._simulation_step()
    gui._update_visualization()
    gui._update_progress_strip()
    gui._update_readouts()
    gui._update_metrics_display()
    gui._update_plots()
    info = gui.sim_last_info
    dt = float(gui.sim_env.unwrapped.dt)

    # Plots are on simulated time, one sample per step (not wall-clock, not per frame).
    assert len(gui.ee_kin_time_hist) == n and len(gui.torque_time_hist) == n
    assert np.allclose(np.diff(gui.ee_kin_time_hist), dt)
    assert gui.sim_time == pytest.approx(n * dt)
    # Velocity is the simulator's hand velocity, torque is the muscles' torque.
    assert gui.ee_vx_hist[-1] == pytest.approx(float(info["ee_velocity"][0]))
    assert gui.latest_shoulder_torque == pytest.approx(float(info["actuator_torque"][0]))
    assert gui.shoulder_gravity_hist[-1] == pytest.approx(float(info["gravity_torque"][0]))
    assert len(gui._ln_vel[0].get_xdata()) == n
    # Every trace fits its plot. The plots were drawn empty first (fixed -1..1),
    # which once switched autoscaling off and clipped everything at +/-1.
    for ax, lines in ((gui.torque_ax, gui._ln_tau), (gui.velacc_ax_vel, gui._ln_vel),
                      (gui.velacc_ax_acc, gui._ln_acc)):
        lo, hi = ax.get_ylim()
        for line in lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            assert lo <= y.min() and y.max() <= hi, (ax.get_title(loc="left"), lo, hi, y.min(), y.max())

    # The readouts say what the simulator says.
    assert gui._tiles["target"].cget("text").endswith("of 3")
    assert gui._tiles["speed"].cget("text") == f"{float(info['ee_speed']):.2f} m/s"
    assert gui._tiles["distance"].cget("text") == f"{100 * float(info['goal_distance']):.1f} cm"
    acts = np.asarray(info["muscle_activations"], dtype=float)
    assert gui._muscle_rows[0][2].cget("text") == f"{100 * acts[0]:.0f}%"
    assert gui._mode_pill.cget("text") == "RUNNING MODEL"

    gui._on_pause()
    assert gui.sim_paused
    gui._on_pause()
    gui._on_next_episode()
    assert gui.sim_episode_count == 2
    gui._stop_simulation()
    gui._update_readouts()
    assert gui._mode_pill.cget("text") == "IDLE"


def test_points_can_be_placed_removed_and_cleared(gui):
    gui._on_clear_points()
    gui._set_place_mode(True)
    assert gui.sim_waypoint_choice == "custom"

    def click(x, y, button=1):
        gui._on_canvas_click(SimpleNamespace(inaxes=gui.ax, xdata=x, ydata=y, button=button))

    click(0.30, 0.25)
    click(0.45, -0.10)
    click(0.10, 0.38)
    assert len(gui.clicked_waypoints) == 3
    assert gui._wp_custom_var.get().count(";") == 2          # the text box lists them
    gui._on_remove_last_point()
    assert len(gui.clicked_waypoints) == 2
    first = np.asarray(gui.clicked_waypoints[0]) - np.asarray(gui._feasibility_env().shoulder_base_position)
    click(float(first[0]), float(first[1]), button=3)         # right-click removes the nearest
    assert len(gui.clicked_waypoints) == 1
    gui._on_clear_points()
    assert gui.clicked_waypoints == [] and gui._wp_custom_var.get() == ""
    gui._set_place_mode(False)


def test_typed_points_become_placed_points(gui):
    gui._wp_custom_var.set("1.40,0.15; 1.20,0.40")
    gui._on_custom_committed()
    assert gui.clicked_waypoints == [[1.40, 0.15], [1.20, 0.40]]
    gui._on_clear_points()


def test_arm_properties_rebuild_the_drawn_arm(gui):
    gui._on_link_length_change(0, 0.45)
    gui._update_visualization()
    assert gui._arm_scene_key[0] == pytest.approx(0.45)
    x = gui._a_arm.get_xdata()
    assert np.hypot(x[1] - x[0], gui._a_arm.get_ydata()[1] - gui._a_arm.get_ydata()[0]) == pytest.approx(0.45)
    from rl_armMotion.two_d.config import ArmConfiguration
    from rl_armMotion.two_d.environments.task_env import DEFAULT_PRESET
    gui.config = ArmConfiguration.get_preset(DEFAULT_PRESET)
    gui._sync_ui_to_config()


def test_green_area_is_the_loaded_models_own_region(gui, tmp_path):
    """Before a run starts, the shaded region comes from the loaded model's
    manifest, not from a default env with the legacy rules."""
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    import json
    human = ArmTaskEnv(random_waypoints=3, joint_margin_deg=5.0, reach_margin=0.0,
                       activation_time_constants=(0.04, 0.06))
    (tmp_path / "run_manifest.json").write_text(json.dumps({"algorithm": "SAC", "env": human.describe()}))
    (tmp_path / "sac_model_best.zip").write_bytes(b"")
    gui._load_model(str(tmp_path / "sac_model_best.zip"))
    assert gui._feasibility_env().joint_margin_deg == 5.0
    assert len(gui._feasible_cloud_points()) == len(human.feasible_points())
    if R3.exists():
        gui._load_model(str(R3))
        assert gui._feasibility_env().joint_margin_deg is None
        assert len(gui._feasible_cloud_points()) == 744
