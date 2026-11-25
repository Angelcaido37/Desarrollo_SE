# Desarrollo de Sistemas Inteligentes para la Toma de Decisiones — EASC-342

**Universidad Autónoma de Nayarit · Licenciatura en Sistemas Computacionales**  
**Responsable académico:** Dr. Oscar Gabriel Vizcaíno Monroy  
**Programa actualizado:** agosto de 2026 · 96 horas · 6 créditos

## Acceso al portal de actividades didácticas

[**Abrir las seis unidades en GitHub Pages**](https://angelcaido37.github.io/Desarrollo_SE/)

## Qué se actualizó

Los **seis laboratorios HTML** de la carpeta `Interactivos/` se rediseñaron como apoyos para las **actividades teóricas de Classroom** de EASC-342. En lugar de exigir programación o despliegue, ahora proporcionan:

- Definiciones concisas, objetivos y cinco tarjetas de repaso por unidad.
- Un simulador didáctico con variables controlables y advertencias sobre alcance.
- Cinco **retos autocorregibles** con pistas, explicación y oportunidad de reintento.
- Seis preguntas de autoevaluación con retroalimentación inmediata.
- Una guía de planificación de la tarea, notas locales y bitácora TXT descargable.

Los datos y escenarios del laboratorio son deliberadamente **diferentes** a los de las tareas evaluables, para evitar sustituir el trabajo individual.

| Unidad | Práctica interactiva | Preparación para Classroom | Ponderación curricular |
|---|---|---|---:|
| 1. Del problema a la arquitectura | [Abrir laboratorio](./Interactivos/Unidad_1_Arquitectura_Decisiones.html) | Arquitectura, AS-IS/TO-BE e indicadores | 15 % |
| 2. Contratos y servicios inteligentes | [Abrir laboratorio](./Interactivos/Unidad_2_Contratos_APIs.html) | Esquemas, idempotencia, timeout y fallback | 15 % |
| 3. Fuentes y respuestas fundamentadas | [Abrir laboratorio](./Interactivos/Unidad_3_RAG_Evidencia.html) | Recuperación, citas, vigencia y abstención | 15 % |
| 4. Automatización con supervisión humana | [Abrir laboratorio](./Interactivos/Unidad_4_Agentes_Aprobaciones.html) | Estados, herramientas, permisos y recuperación | 15 % |
| 5. Interfaces que ayudan a decidir | [Abrir laboratorio](./Interactivos/Unidad_5_Interfaces_Usabilidad.html) | Jerarquía visual, incertidumbre y accesibilidad | 15 % |
| 6. Operación, incidentes y liberación | [Abrir laboratorio](./Interactivos/Unidad_6_Observabilidad_Rollback.html) | SLO, logs, seguridad y rollback | 25 % |

## Cómo estudiar

1. Abre la unidad correspondiente desde el `index.html`.
2. Estudia los conceptos y marca las tarjetas que necesites volver a repasar.
3. Resuelve el caso alternativo usando el simulador y después los cinco retos.
4. Comprueba tus conocimientos con seis preguntas de opción múltiple.
5. En **Mi tarea**, completa tu plan personal y desarrolla la evidencia solicitada en Google Classroom, **sin copiar los resultados del caso alternativo**.
6. Descarga la bitácora local si deseas conservar un registro de avances.

## Alcance y límites

Los HTML están hechos con CSS y JavaScript integrados, **sin dependencias ni conexión a servicios externos**. Guardan el avance únicamente en el navegador que se utilice, mediante `localStorage`, y permiten exportar notas a TXT. No se envían calificaciones a Google Classroom y no se recopilan datos personales.

Los simuladores no constituyen APIs reales, flujos productivos, RAG generativo, agentes con herramientas conectadas ni despliegues. Esta adaptación permite apoyar la **cobertura teórica impartida**. No demuestra las competencias de integración y operación exigidas por el programa original, que deben evaluarse conforme a la adecuación académica correspondiente.

## Evidencias curriculares teóricas adaptadas

- **I (15 %):** especificación del problema, AS-IS/TO-BE, línea base y arquitectura.
- **II (15 %):** análisis de un contrato de inferencia y manejo conceptual de errores.
- **III (15 %):** fuentes, citas, vigencia, contradicción e inyección en RAG.
- **IV (15 %):** permisos, estados, cinco pruebas de escritorio y escalamiento.
- **V (15 %):** bocetos, accesibilidad, prueba real con dos compañeros y ajustes.
- **VI (25 %):** auditoría de diez o menos registros del caso, métricas, incidentes, rollback y defensa.

La **tarea de Classroom** sigue siendo el instrumento evaluable; el laboratorio es una preparación, con preguntas y ejemplos distintos. El proyecto operable del programa y su defensa tienen requisitos curriculares independientes, por lo que su sustitución exige la justificación institucional correspondiente.

## Publicación en GitHub Pages

La rama `main` contiene `index.html`, `.nojekyll`, `README.md`, `Interactivos/` y `Material/` (recursos históricos conservados). El flujo en `.github/workflows/publicar-laboratorios.yml` comprueba los seis archivos antes de desplegar la web.

## Referencias curriculares

- Programa de estudio EASC-342, *Desarrollo de Sistemas Inteligentes para la Toma de Decisiones*, actualización de agosto de 2026.
- NIST (2023). *AI Risk Management Framework (AI RMF 1.0)*. https://doi.org/10.6028/NIST.AI.100-1