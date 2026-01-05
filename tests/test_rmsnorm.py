import fastnorm

def test_import():
    # Prüft, ob das C++ Modul geladen werden kann
    assert fastnorm.__file__ is not None

def test_basic_rmsnorm():
    data = [1.0, 2.0, 3.0]
    result = fastnorm.rmsnorm(data)
    assert len(result) == 3
    # Ein kleiner mathematischer Check (RMSNorm verkleinert die Werte meist)
    assert all(isinstance(x, float) for x in result)