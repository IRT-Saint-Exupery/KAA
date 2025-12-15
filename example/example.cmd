;================================================
; RENAULT WELDING methodes
;================================================

;----------USE CASE----------
k:0
c:'UCRenaultWeldingResnet
; model
o:'Resnet
; data
dm:1 dn:2

;----------EXPLAINABILITY----------
k:1
;       LIBRARY
b:'Xplique

;       METHODS
h:'Occlusion
;       PARAMETRES
xobz:64 xopz:35 xops:3 xoov:1

;----------REPORTS----------
k:2
tw:1 tl:1

;----------EXECUTION----------
k:3

;       ACTIONS
; remove all
a:d
; selection
a:0 a:1 a:2 a:5

;       MANAGEMENT
; log file
L:RWr_o
; path for results
P:exampleRWr
; trace
T:1

;       RUN
l

;----------EXPLAINABILITY----------
k:1
;       METHODS
h:'Rise
;       PARAMETRES
xrns:2000 xrgz:7 xrpp:0.5

;----------EXECUTION----------
k:3

;       MANAGEMENT
; log file: different
L:RWr_r

;       RUN
l

;----------EXPLAINABILITY----------
k:1
;       METRICS
m:'CausalFidelity-Deletion
;       PARAMETRES
xdst:100
;       COMPUTE ON
; remove all
u:d
; selection
u:3 u:4

;----------REPORTS----------
k:2
;       FULL REPORT
; on explanations and metrics
f:0
f:1

;----------EXECUTION----------
;k:3
;       ACTIONS
; invert actions
a:i

;       MANAGEMENT
; log file: different
L:RWr_m

;       RUN
l

;================================================
X
