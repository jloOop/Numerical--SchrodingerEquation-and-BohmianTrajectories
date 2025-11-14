Assets (animations and example outputs)

Please place animations and large outputs in this directory.

Filename conventions:
- Use safe filenames without special characters, e.g.:
  bohmian_animation_kappa_1.gif
  bohmian_animation_kappa_1.mp4

If you plan to add many or large files, install Git LFS locally and track media types before pushing:
- git lfs install
- git lfs track "*.gif"
- git lfs track "*.mp4"
- git add .gitattributes
- git commit -m "Track animation media with Git LFS"

To add the GIF locally:
1) copy the GIF into assets/, e.g.:
   mv /path/to/bohmian_animation_kappa=1.gif assets/bohmian_animation_kappa_1.gif

2) commit and push:
   git add assets/bohmian_animation_kappa_1.gif
   git commit -m "Add bohmian animation kappa=1"
   git push origin add-bohmian-example

Alternatively, if you upload the GIF here or provide a public URL, I can add it to the branch for you.
