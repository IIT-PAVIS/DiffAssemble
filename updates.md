# Puzzle Diffusion

## Supporting idea
Diffusion models solve problems of recovering $x_0$ by "denoising" a noisy data $x_T$ with a chain process such that $x_0=p(x_T)\prod^T{p(x_{t-1}|x_t)}$

For images, typically $x_0$ is the image, and $x_T$ is noise sampled from $\mathcal{N}(0,1)$.

In our case we want to recover position of 2D patches of in image. So $p_\theta(pos_t | pos_{t-1},rgb)$.

---
## TODO
- [x] Add DDIM, with skippable steps
- [ ] Test rotations
- [ ] Test missing pieces
- [ ] **
## Updates:

**01/02/2023**
- New Test idea: Reconstruct missing patches
- paper to compare? Solving Jigsaw Puzzles With Eroded Boundaries
Dov Bridger, Dov Danon, Ayellet Tal; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3526-3535
 https://openaccess.thecvf.com/content_CVPR_2020/html/Bridger_Solving_Jigsaw_Puzzles_With_Eroded_Boundaries_CVPR_2020_paper.html



**31/01/2023**
- Implemented DDIM sampling-> sampling speedup
- Tested the GAT as GNN on wikiart 12x12, doesn't work well
