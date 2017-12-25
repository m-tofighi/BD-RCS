# BD-RCS
Blind Image Deblurring Using Row-Column Sparse Representations

Blind image deblurring is a particularly challenging inverse problem where the blur kernel is unknown and must be estimated en route to recover the deblurred image. The problem is of strong practical relevance since many imaging devices such as cellphone cameras, must rely on deblurring algorithms to yield satisfactory image quality. Despite significant research effort, handling large motions remains an open problem. In this paper, we develop a new method called Blind Image Deblurring using Row-Column Sparsity (BD-RCS) to address this issue. Specifically, we model the outer product of kernel and image coefficients in certain transformation domains as a rank-one matrix, and recover it by solving a rank minimization problem. Our central contribution then includes solving {\em two new} optimization problems involving row and column sparsity to automatically determine blur kernel and image support sequentially. The kernel and image can then be recovered through a singular value decomposition (SVD). Experimental results on linear motion deblurring demonstrate that BD-RCS can yield better results than state of the art, particularly when the blur is caused by large motion. This is confirmed both visually and through quantitative measures.

In order to solve the optimization problem in Eq. 5 and Eq. 6 in the paper we used minFunc. For convenience I also added that pachage to this repository.

Start from "BDRCS_demo.m" and from there you can find the path to other main fucntions.

Best,
M. Tofighi
