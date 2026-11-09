Scripts/build_app.sh --install

for p in TurboFieldfareCLI TurboFieldfareServer TurboFieldfareRepack; do swift build -c release --product "$p" || break; done

LIB="$HOME/.local/lib/turbofieldfare"
BIN="$(swift build -c release --show-bin-path)"
mkdir -p "$LIB" "$HOME/.local/bin"
cp "$BIN"/TurboFieldfareCLI "$BIN"/TurboFieldfareServer "$BIN"/TurboFieldfareRepack "$LIB"/
for b in "$BIN"/*.bundle; do ditto "$b" "$LIB/$(basename "$b")"; done
for t in TurboFieldfareCLI TurboFieldfareServer TurboFieldfareRepack; do
  printf '#!/bin/sh\nexec "%s/%s" "$@"\n' "$LIB" "$t" > "$HOME/.local/bin/$t"
  chmod +x "$HOME/.local/bin/$t"
done

rm -rf .build dist
