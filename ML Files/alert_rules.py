"""Alert rule applied after the model (a fixed, explainable rule, not learned).

A THREAT call from the model only becomes a THREAT ALARM when both are true:
  1. Drone confirmed: positive evidence it is a drone, not a bird:
       camera identifies a drone (3+ looks, 2+ of them saying "drone" from inside identification range, 60% of camera range), OR radar sees a rotor signature (micro-Doppler) on 60% of 10+ recent returns and the camera has not identified a bird,
       OR a drone control link / Remote ID is received (unless a radar with a clear view of that spot has never seen anything there).
  2. Urgent: over the bowl, OR within CLOSE_M of midfield, OR it would reach the bowl within URGENT_S.
Otherwise the operator sees BORDERLINE ("possible threat, confirming") and the camera queue gives the
track priority. Nothing is dropped: an unconfirmed track is still on screen and still alerts the operator.
"""
import os
URGENT_S = float(os.environ.get("URGENT_S", 120))      # seconds to the bowl
CLOSE_M = float(os.environ.get("CLOSE_M", 600))       # metres from midfield


MD_MIN_RETURNS = 10  # rotor signature must hold over at least this many recent radar returns...
MD_MIN_SHARE = 0.6   # ...on at least 60% of them (bird wing beats sometimes look rotor-like for a few sweeps)
CAM_MIN_LOOKS = int(os.environ.get("CAM_MIN_LOOKS", 3))     # camera must have looked this many times before its "drone" call confirms
CAM_OVERTURN_LOOKS = int(os.environ.get("CAM_OVERTURN_LOOKS", 4))  # a later camera "bird" call with this many looks withdraws an earlier confirmation


RADIO_GHOST_S = int(os.environ.get("RADIO_GHOST_S", 5))


def confirmed(f):
    radio = f["rf_frac"] > 0.1 or f["rid_seen"]             # a drone control link or Remote ID: birds never transmit
    # ghost check: a radio signal alone does not confirm a drone if a radar has a clear view of that spot and has never seen
    # anything there (a phone hotspot or Wi-Fi device that resembles a drone link). The camera or the radar must confirm.
    # While a radar with a clear view has had fewer than 5 s to look, the radio signal waits for it (a real drone in view of
    # a radar is usually picked up within a second or two); where no radar can see the spot, radio confirms on its own.
    if radio and f["radar_frac"] == 0 and f.get("radar_opp", 0) > 0: radio = False
    camera = f["cam_drone"] and f["cam_looks"] >= CAM_MIN_LOOKS and f.get("cam_drone_id_looks", 2) >= 2   # 2+ "drone" looks from inside identification range
    rotor = f["md_returns"] >= MD_MIN_RETURNS and f["md_hits"] >= MD_MIN_SHARE * f["md_returns"]
    bird_seen = f["cam_bird"] and f["cam_looks"] >= 2
    # camera veto: if the camera has identified a bird and nothing is transmitting, a rotor-like radar return alone does not confirm a drone
    return bool(radio or camera or (rotor and not bird_seen))


def update_latch(latched, f):
    """Confirmation is kept for the life of the track, with one exception: if nothing is transmitting and the camera,
    after more looks, now identifies a bird, the earlier confirmation is withdrawn (a camera at the edge of its range
    can be wrong on its first few looks; operators re-check the same way)."""
    radio = f["rf_frac"] > 0.1 or f["rid_seen"]
    if not radio and f["cam_bird"] and f["cam_looks"] >= CAM_OVERTURN_LOOKS: return False
    return bool(latched or confirmed(f))


def urgent(f):
    return bool(f["over_bowl"] or f["dist_m"] <= CLOSE_M or f["time_to_bowl_s"] <= URGENT_S)


def gate(label, f, was_confirmed=False):
    """was_confirmed: this track was confirmed as a drone earlier (confirmation is kept for the life of the track)."""
    if label == "threat" and not ((was_confirmed or confirmed(f)) and urgent(f)):
        return "borderline"
    return label


LEVELS = ["none", "watch", "alert", "alarm"]


def level(label, f, was_confirmed=False):
    """What the operator gets for a track this second.
      alarm: threat, confirmed drone and urgent (the gate above already guarantees both)
      alert: operator notified (borderline or careless, when it is a confirmed drone or urgent)
      watch: on screen, no notification (unconfirmed and not urgent: e.g. a far-off bird flock)
      none:  authorized or not a drone"""
    if label in (None, "authorized", "non_drone"): return "none"
    if label == "threat": return "alarm"
    return "alert" if (was_confirmed or confirmed(f) or urgent(f)) else "watch"
