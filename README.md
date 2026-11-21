# Desarrollo de Sistemas Inteligentes para la Toma de Decisiones (EASC-342)

**Universidad Autónoma de Nayarit** · Área Económico Administrativo · Licenciatura en Sistemas Computacionales  
**Responsable académico:** Dr. Oscar Gabriel Vizcaíno Monroy  
**Academia:** Tratamiento de la Información · **Actualización curricular:** agosto de 2026  
**Modalidad:** curso-taller obligatorio · **6 créditos** · **96 horas** (16 teóricas, 36 prácticas y 44 independientes).

## Acceder al portal

**[Abrir los seis laboratorios interactivos](https://angelcaido37.github.io/Desarrollo_SE/)**

Este repositorio presenta actividades formativas en HTML para aprender a **desarrollar, integrar, desplegar, probar y operar** sistemas inteligentes que apoyan la toma de decisiones. A diferencia de una introducción a la IA, aquí el foco está en contratos de componentes, servicios y APIs, RAG, herramientas y agentes, UX para decisiones, observabilidad, seguridad, aprobación humana y recuperación ante fallos.

Los laboratorios se articulan mediante el caso ficticio **CampusDecide**, un sistema universitario de apoyo a decisiones. No representa trámites, horarios ni datos oficiales de la UAN.

## Unidades, laboratorios y ponderación

| Unidad | Contenidos principales | Valor | Actividad |
|---|---|---:|---|
| I. Problema y arquitectura | Procesos AS-IS/TO-BE, líneas base, requisitos, responsabilidades e indicadores. | 15 % | [Abrir HTML](./Interactivos/Unidad_1_Arquitectura_Decisiones.html) |
| II. Contratos e integración | Validación de esquemas, API, errores, timeout, idempotencia, reintentos y fallback. | 15 % | [Abrir HTML](./Interactivos/Unidad_2_Contratos_APIs.html) |
| III. Aplicaciones RAG y evidencia | Fuentes, vigencia documental, recuperación, citas, abstención y trazabilidad. | 15 % | [Abrir HTML](./Interactivos/Unidad_3_RAG_Evidencia.html) |
| IV. Agentes y automatización | Herramientas, acciones autorizadas, aprobación humana, límites y auditoría. | 15 % | [Abrir HTML](./Interactivos/Unidad_4_Agentes_Aprobaciones.html) |
| V. Interfaces de decisión | Recomendaciones explicables, confianza, accesibilidad y prueba con usuarios. | 15 % | [Abrir HTML](./Interactivos/Unidad_5_Interfaces_Usabilidad.html) |
| VI. Despliegue y operación | Observabilidad, seguridad, latencia, incidentes, versiones y rollback. | 25 % | [Abrir HTML](./Interactivos/Unidad_6_Observabilidad_Rollback.html) |

Cada laboratorio ofrece conceptos y tarjetas de repaso, simuladores técnicos, seis preguntas con retroalimentación inmediata y apoyo para preparar la tarea. Los puntos obtenidos son **formativos**, no una calificación curricular oficial.

## Resultados de aprendizaje

Los estudiantes deberán poder convertir una necesidad de decisión en un proceso y arquitectura verificables; integrar componentes y modelos previamente validados; construir aplicaciones fundamentadas, agentes o automatizaciones supervisadas; diseñar interfaces accesibles; desplegar un prototipo; registrar métricas, costos y fallos; y aplicar medidas de recuperación, trazabilidad y gobernanza.

## Evidencias curriculares

1. Especificación del problema, proceso y arquitectura (15 %).
2. Contrato de inferencia e integración de componente (15 %).
3. Aplicación RAG con fuentes y evaluación (15 %).
4. Agente o automatización con supervisión humana (15 %).
5. Interfaz de apoyo a decisiones y prueba con usuarios (15 %).
6. Proyecto integrador operable y defensa individual (25 %).

La acreditación del programa requiere un mínimo de 80 % de asistencia, calificación final de 60/100 y **aprobar por separado** el proyecto integrador y la defensa individual con al menos 60/100 en cada componente. También exige demostrar autoría y declarar el uso de herramientas de IA.

## Organización del repositorio

```text
Desarrollo_SE/
├── index.html
├── README.md
├── .nojekyll
├── Interactivos/
│   ├── Unidad_1_Arquitectura_Decisiones.html
│   ├── Unidad_2_Contratos_APIs.html
│   ├── Unidad_3_RAG_Evidencia.html
│   ├── Unidad_4_Agentes_Aprobaciones.html
│   ├── Unidad_5_Interfaces_Usabilidad.html
│   ├── Unidad_6_Observabilidad_Rollback.html
└── Material/  # Recursos anteriores; se conservan
```

## Uso en el aula

Abre el [portal](https://angelcaido37.github.io/Desarrollo_SE/), selecciona una unidad y completa las simulaciones. Para Google Classroom, adjunta el enlace del laboratorio correspondiente y solicita las evidencias, los cálculos, la bitácora y la justificación de decisiones que defina la tarea.

**Alcance:** los seis HTML son autónomos y funcionan como simuladores de navegador. No realizan llamadas a APIs productivas ni despliegues reales. El laboratorio de la Unidad III representa recuperación documental extractiva, **no un sistema RAG conectado a un LLM**. No se deben ingresar secretos ni datos personales reales. Ningún resultado de una simulación sustituye pruebas de integración, seguridad o validación con usuarios.

## Publicación con GitHub Pages

En **Settings → Pages**, utiliza **Deploy from a branch**, rama `main`, carpeta `/(root)`. El archivo `.nojekyll` evita el procesamiento de Jekyll; los enlaces del sitio son relativos al proyecto. Si el despliegue falla después de compilar y generar el artefacto correctamente, revisa el estado de GitHub Pages y Actions antes de cambiar el código.

## Referencias fundamentales

- Huyen, C. (2025). *AI Engineering: Building Applications with Foundation Models*. O'Reilly.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly.
- Reis, J. y Housley, M. (2022). *Fundamentals of Data Engineering*. O'Reilly.
- NIST (2023). *AI Risk Management Framework (AI RMF 1.0)*. https://doi.org/10.6028/NIST.AI.100-1

**Nota:** los recursos actuales tienen fines académicos y no constituyen sistemas productivos de apoyo a decisiones.
