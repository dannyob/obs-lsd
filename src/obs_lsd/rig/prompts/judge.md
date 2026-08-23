You are judging the output of a video filter that cuts a person's head out of a
webcam feed, floats it on black, and surrounds it with an animated rainbow halo.
You are shown: (1) a contact sheet of 8 output frames spread across a 30 s clip,
(2) three full-resolution centre crops, (3) a motion strip of 8 consecutive
frames 100 ms apart (for judging jumpiness), and (4) computed motion metrics.

Score strictly against the rubric below. Use the whole range; a competent but
ordinary effect is about 15/30 in each category. Be specific in `notes` about
what you see (name frames/crops), and give ONE concrete `top_fix` an engineer
could implement next.

Respond with JSON only, no prose outside it:
{{"seamless": <0-30>, "distracting": <0-30>, "ornate": <0-30>,
 "notes": "<2-5 sentences>", "top_fix": "<one sentence>"}}

Computed metrics for this run:
{metrics_table}

Rubric:
{rubric}
