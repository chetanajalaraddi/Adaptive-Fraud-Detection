from river import drift


def test_adwin_initialization():

    detector = drift.ADWIN()

    assert detector is not None


def test_adwin_accepts_observations():

    detector = drift.ADWIN()

    for _ in range(100):
        detector.update(0)

    assert detector.width > 0


def test_adwin_detects_change():

    detector = drift.ADWIN()

    for _ in range(200):
        detector.update(0)

    drift_detected = False

    for _ in range(200):
        detector.update(1)

        if detector.drift_detected:
            drift_detected = True
            break

    assert drift_detected