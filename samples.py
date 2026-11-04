"""Questions that exercise multi-step tool use, memory, and a failed lookup.

Every question in a demo run shares one thread_id. A follow-up omits the
subject on purpose: the checkpoint still has the previous turn.
"""

DEMO_QUESTIONS: list[tuple[str, str]] = [
    (
        "multi-step",
        "Para la competencia Empatía, ¿qué videos ya validó un humano y qué dice la transcripción del primero?",
    ),
    (
        "memory",
        "¿Y para Liderazgo?",
    ),
    (
        "unknown competence",
        "¿Qué assets cubren la competencia Resiliencia?",
    ),
    (
        "incomplete asset",
        "¿El asset El silencio en la mesa está clasificado y qué dice su transcripción?",
    ),
    (
        "trainings",
        "¿Qué entrenamientos tiene asignados ana@nimbus.test?",
    ),
    (
        "training memory",
        "¿Y cuál es su progreso en Empatía?",
    ),
]
