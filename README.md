# Magic Frame — HANDSTORM v13.1

Ta wersja nie modyfikuje obrazu wewnątrz wielokąta. Kamera pozostaje oryginalna, a cała magia jest rysowana jako reaktywna warstwa animacji oparta o pełny tracking dłoni.

## Co robi v13
- MediaPipe Hand Landmarker: 21 landmarków na każdą z maks. 2 dłoni.
- Filtr One Euro per landmark: mniej jitteru bez odczuwalnego opóźnienia.
- Krótka predykcja pozycji na podstawie prędkości punktów.
- Stabilna identyfikacja dwóch dłoni na podstawie pozycji nadgarstków.
- Smugi każdego z 5 palców, cząsteczki, glow i energetyczny szkielet dłoni.
- Pinch (kciuk + wskazujący): singularity + radial shockwave + burst particles.
- Szybki ruch: gęstsze smugi i iskry.
- Zaciśnięcie pięści ładuje energię; szybkie otwarcie odpala mocny radialny burst.
- Dwie dłonie: dynamiczny portal z pierścieniami i łukami pomiędzy palcami.
- 3 style: Quantum Rift, Solar Flare, Void Storm.
- Regulacja mocy efektu 55–145%.
- Debug trackingu oraz nagrywanie finalnego canvasu.

## Start
```bash
npm install
npm run dev
```

Do kamery wymagany jest HTTPS albo localhost.


## v13.1 reliability fix
- MediaPipe loads LOCAL first and automatically falls back to jsDelivr + Google model CDN.
- Hand tracker initializes before the camera starts.
- 9s init timeout exposes stuck loading instead of remaining on `load…`.
- Lowered detection thresholds for easier first-lock.
