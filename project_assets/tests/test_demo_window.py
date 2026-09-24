"""The demo window: every control present, and the point editing it advertises.

The full controller's buttons were built correctly but clipped out of view by a
short pane, which is not something a unit test had any way to notice. These
tests therefore check reachability as well as behaviour: the controls exist, are
mapped into the window, and sit inside it at a normal size.
"""

import numpy as np
import pytest

from rl_armMotion.two_d.evaluation import make_env
from rl_armMotion.two_d.gui.waypoint_editing import REMOVE_RADIUS, index_near, nearest_placeable


# --------------------------------------------------------------------------- #
# Placement rules (no display needed)
# --------------------------------------------------------------------------- #

def test_nearest_placeable_moves_an_unreachable_click_inward():
    env = make_env(actuation_mode="muscle", waypoints=None)
    shoulder = np.asarray(env.shoulder_base_position, dtype=float)
    far = [float(shoulder[0] + 0.98 * env.reach), float(shoulder[1])]
    assert not env.waypoint_feasibility(far, env.max_curriculum_tolerance).get("feasible", False)

    snapped, moved = nearest_placeable(env, far, [])
    assert snapped is not None and moved > 0.0
    assert env.waypoint_feasibility(snapped, env.max_curriculum_tolerance)["feasible"]
    assert np.linalg.norm(np.asarray(snapped) - shoulder) < np.linalg.norm(np.asarray(far) - shoulder)


def test_nearest_placeable_leaves_a_good_click_alone():
    env = make_env(actuation_mode="muscle", waypoints=None)
    good = [float(v) for v in env.feasible_points()[0]]
    same, moved = nearest_placeable(env, good, [])
    assert moved == pytest.approx(0.0, abs=1e-9) and same == pytest.approx(good)


def test_index_near_picks_the_point_under_the_cursor_or_nothing():
    pts = [[1.15, 0.33], [1.42, 0.08], [1.30, -0.30]]
    assert index_near(pts, [1.42, 0.08]) == 1
    assert index_near(pts, [1.42 + 0.5 * REMOVE_RADIUS, 0.08]) == 1     # near enough
    assert index_near(pts, [1.00, 0.60]) is None                        # open space
    assert index_near([], [1.2, 0.2]) is None


# --------------------------------------------------------------------------- #
# The window itself
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo():
    """One window for the module: a Tk root per test fails on the third."""
    tk = pytest.importorskip("tkinter")
    from rl_armMotion.two_d.gui.demo_app import DemoWindow
    win = DemoWindow(points=None, autostart=False)
    try:
        root = win.build()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display available: {exc}")
    root.update_idletasks()
    root.update()          # winfo_ismapped() is only meaningful once realised
    yield win
    win.root.destroy()


def test_every_control_exists_and_is_mapped(demo):
    """The failure mode this window was built to end: a control that exists in
    the code but never reaches the screen."""
    controls = {
        "add point": demo.add_btn, "remove point": demo.del_btn,
        "start": demo.start_btn, "stop": demo.stop_btn,
        "speed": demo.speed_scale, "hold bar": demo.hold_bar,
        "points hint": demo.points_hint, "model label": demo.model_label,
        "status": demo.status, "reached": demo.reached_label,
    }
    for name, widget in controls.items():
        assert widget.winfo_exists(), f"{name} does not exist"
        assert widget.winfo_ismapped(), f"{name} exists but is not on screen"


def test_controls_fit_inside_the_window_at_its_minimum_size(demo):
    demo.root.geometry("1120x700")
    demo.root.update_idletasks()
    demo.root.update()
    w, h = demo.root.winfo_width(), demo.root.winfo_height()
    for name, widget in (("add", demo.add_btn), ("remove", demo.del_btn),
                         ("start", demo.start_btn), ("stop", demo.stop_btn),
                         ("hold bar", demo.hold_bar)):
        right = widget.winfo_rootx() - demo.root.winfo_rootx() + widget.winfo_width()
        bottom = widget.winfo_rooty() - demo.root.winfo_rooty() + widget.winfo_height()
        assert right <= w + 1, f"{name} is cut off the right edge ({right} > {w})"
        assert bottom <= h + 1, f"{name} is cut off the bottom ({bottom} > {h})"


def test_adding_and_removing_points_through_the_click_handler(demo):
    from types import SimpleNamespace
    demo.points, demo.requested = [], []
    demo.mode_var.set("add")
    demo._on_mode()

    def click(x, y, button=1):
        demo._on_click(SimpleNamespace(inaxes=demo.ax, xdata=x, ydata=y, button=button))

    for p in [(1.15, 0.33), (1.42, 0.08), (1.30, -0.30)]:
        click(*p)
    assert len(demo.points) == 3

    demo.mode_var.set("remove")
    demo._on_mode()
    click(1.42, 0.08)
    assert len(demo.points) == 2
    assert all(abs(p[0] - 1.42) > 1e-6 for p in demo.points), "deleted the wrong point"
    click(1.00, 0.60)                       # open space removes nothing
    assert len(demo.points) == 2

    demo.mode_var.set("add")
    demo._on_mode()
    click(1.40, 0.05)
    assert len(demo.points) == 3

    demo._undo()
    assert len(demo.points) == 2
    demo._clear()
    assert demo.points == [] and demo.requested == []


def test_switching_to_my_own_points_empties_the_demo_set(demo):
    demo.set_var.set("demo")
    demo._on_set_choice()
    assert len(demo.points) == 3 and demo.mode == "off"

    demo.set_var.set("mine")
    demo._on_set_choice()
    assert demo.points == [] and demo.mode == "add"


def test_hint_tracks_the_mode_and_the_count(demo):
    demo.points, demo.requested = [], []
    demo.mode_var.set("add")
    demo._on_mode()
    assert "Click the picture" in demo.points_hint.cget("text")
    demo.points = [[1.2, 0.3]]
    demo.mode_var.set("remove")
    demo._on_mode()
    assert "delete" in demo.points_hint.cget("text")
    demo.mode_var.set("off")
    demo._on_mode()
    assert "Start" in demo.points_hint.cget("text")
