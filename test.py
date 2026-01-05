import fastnorm

daten = [1.0, 2.0, 3.0, 4.0]
daten = fastnorm.rmsnorm(daten)
print(daten)