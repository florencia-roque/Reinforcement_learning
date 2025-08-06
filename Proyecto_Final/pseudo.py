for i = 0 to escenarios:
    h=inicializarHidrologia(hidroIni)
    escenario[i] = nuevoEscenario
    for j = 0 to 51:
        agregarSemana(escenario[i],aportes(h,j))
        h=siguienteHidrologia
