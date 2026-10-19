# Pull the accepted zoom movie to the Mac

Run on the Mac. This pulls only final review artifacts, preserving older movie
folders. Use your normal thin-mirror root if it differs from the example.

```sh
remote=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2
movie=jeans_session010_particles3d_t0_to98_zoom.mp4
dest="$HOME/Desktop/Research/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2"
mkdir -p "$dest"
scp "jiaxiwu@perseus.caltech.edu:$remote/$movie" "jiaxiwu@perseus.caltech.edu:$remote/$movie.sha256" "jiaxiwu@perseus.caltech.edu:$remote/CAPTION.md" "$dest/"
(cd "$dest" && shasum -a256 -c "$movie.sha256")
```

Expected SHA256:`70220846f958d2067143f313eab637486278460bfaadb9e925b231876fc91556`. Expected bytes:6300268.
Do not copy frames, particle caches or sequence links. No Mac copy was performed
by the Perseus agent, and no cluster original was deleted.
