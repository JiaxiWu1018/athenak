# Verified one-way local pull

Run these commands **on the Mac**, not on Perseus. They copy only final review
artifacts into a new versioned movie directory and verify the MP4. Adapt only
the local destination if your thin mirror has a different root.

```sh
remote=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1
movie=jeans_session010_particles3d_t0_to98.mp4
dest="$HOME/Desktop/Research/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1"
mkdir -p "$dest"
scp "jiaxiwu@perseus.caltech.edu:$remote/$movie" "jiaxiwu@perseus.caltech.edu:$remote/$movie.sha256" "jiaxiwu@perseus.caltech.edu:$remote/CAPTION.md" "$dest/"
(cd "$dest" && shasum -a 256 -c "$movie.sha256")
```

Expected movie SHA256:
`724d73b0508efa8c81fdd025203770d150fd1ed2ff91ad888da181b305f8ae22`.
Expected size:5,005,907bytes. Do not remove cluster originals after this pull.
This script has not been executed against a Mac by the Perseus agent.
