from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza


@registra("cl", set(), "clustering delle coordinate", REGISTRO)
@monitora("cl")
def cl(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[int] | None:
    coordinate = dati.dati.xy
    cluster = miomodulo.clustering(impostazioni["icl"], dati.dati.taglia, coordinate[:, 0].tolist(),
                                   coordinate[:, 1].tolist(), impostazioni["ikcl"], impostazioni["cmcl"],
                                   impostazioni["scl"], impostazioni["cbcl"], impostazioni["abcl"],
                                   impostazioni["iicl"], impostazioni["iecl"], impostazioni["ncl"],
                                   impostazioni["acl"], impostazioni["sccl"], impostazioni["cscl"],
                                   impostazioni["bicl"], impostazioni["fcl"])
    return cluster
