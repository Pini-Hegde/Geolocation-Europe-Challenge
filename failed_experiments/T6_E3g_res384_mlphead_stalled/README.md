# T6_E3g — res384 + MLP country head + proj160: STALLED, killed ep8

3 changes bundled (input 384, country_head_hidden 192, cell_proj_dim 160). Trajectory:
ep5 med 230 ctry 0.727 -> ep8 med 234 ctry 0.711 (BACKWARDS). E3e ep8: med 189 ctry 0.744.
384px+batch16 noisier, MLP head worse here, reduced cell proj weaker. Resolution route abandoned.

Country ceiling (~0.78) needs: (a) country-gated decode on E3e ckpt (CHEAP - do first), or
(b) mobilenetv4_conv_small backbone (2.5M trunk) + direct big heads.
