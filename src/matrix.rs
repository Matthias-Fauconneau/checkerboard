use std::array::from_fn as eval;
pub fn transpose<T: Copy, const M: usize, const N:usize>(a: [[T; N]; M]) -> [[T; M]; N] { eval(|i| eval(|j| a[j][i])) }

#[allow(non_camel_case_types)] type Matrix<const M: usize, const N:usize> = [[f32; N]; M];
#[allow(non_camel_case_types)] pub type mat3 = Matrix<3,3>;

pub fn identity<const M: usize, const N:usize>() -> [[f32; M]; N] { eval(|i| eval(|j| if i==j {1.} else {0.})) }
pub fn mul<const M: usize, const N:usize, const P:usize>(a: Matrix<M,N>, b: Matrix<N,P>) -> Matrix<M,P> { eval(|i| eval(|j| (0..N).map(|k| a[i][k]*b[k][j]).sum())) }

use vector::vec2;
pub fn apply(M: mat3, v: vec2) -> vec2 { eval(|i| v.x*M[i][0]+v.y*M[i][1]+M[i][2]).into() }

fn det(M: mat3) -> f32 {
	let M = |i: usize, j: usize| M[i][j];
	M(0,0) * (M(1,1) * M(2,2) - M(2,1) * M(1,2)) -
	M(0,1) * (M(1,0) * M(2,2) - M(2,0) * M(1,2)) +
	M(0,2) * (M(1,0) * M(2,1) - M(2,0) * M(1,1))
}
fn cofactor(M: mat3) -> mat3 { let M = |i: usize, j: usize| M[i][j]; [
	[(M(1,1) * M(2,2) - M(2,1) * M(1,2)), -(M(1,0) * M(2,2) - M(2,0) * M(1,2)),   (M(1,0) * M(2,1) - M(2,0) * M(1,1))],
	[-(M(0,1) * M(2,2) - M(2,1) * M(0,2)),   (M(0,0) * M(2,2) - M(2,0) * M(0,2)),  -(M(0,0) * M(2,1) - M(2,0) * M(0,1))],
	[(M(0,1) * M(1,2) - M(1,1) * M(0,2)),  -(M(0,0) * M(1,2) - M(1,0) * M(0,2)),   (M(0,0) * M(1,1) - M(0,1) * M(1,0))],
] }
fn adjugate(M: mat3) -> mat3 { transpose(cofactor(M)) }
fn scale(s: f32, M: mat3) -> mat3 { M.map(|row| row.map(|e| s*e)) }
pub fn inverse(M: mat3) -> mat3 { scale(1./det(M), adjugate(M)) }
pub fn direct_linear_transform(P: [[vec2; 4]; 2]) -> mat3 {
	/*let X = [X.map(|p| p.x), X.map(|p| p.y), X.map(|_| 1.)];
	let Y = [Y.map(|p| p.x), Y.map(|p| p.y), Y.map(|_| 1.)];
	[[0., 0., 0., -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
	[z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]]
	mul(mul(X, transpose(Y)), inverse(mul(Y, transpose(Y))))*/
	let mut AtA = nalgebra::OMatrix::<f32, nalgebra::U9, nalgebra::U9>::zeros();// [[0.; 9]; 9];
	for [P,p] in transpose(P) {
		let x = [P.x,P.y,1., 0.,0.,0., -p.x*P.x, -p.x*P.y, -p.x];
		let y = [0.,0.,0., P.x,P.y,1., -p.y*P.x, -p.y*P.y, -p.y];
		for i in 0..9 { for j in 0..=i { AtA[(i,j)/*i][j*/] += x[i]*x[j] + y[i]*y[j]; } } // Lower triangle
		//for j in 0..9 { for i in j..9 { AtA[(i,j)/*i][j*/] += x[i]*x[j] + y[i]*y[j]; } } // Lower triangle
	}
	dbg!(AtA);
	//for i in 0..9 { for j in i+1..9 { AtA[i][j] = AtA[j][i]; } }
	//dbg!(AtA);
	let Ql = nalgebra::linalg::SymmetricEigen::new(AtA);
	let (Q, eigenvalues) = (Ql.eigenvectors, Ql.eigenvalues);
	//assert!(eigenvalues[0] == eigenvalues.min(), "{eigenvalues}");
	println!("{Q} {eigenvalues} {}", eigenvalues.iter().enumerate().min_by(|&(_, &a), &(_, b)| a.total_cmp(b)).unwrap().0);
	//eigenvalues.iter().enumerate().min_by(|&(_, &a), &(_, b)| a.total_cmp(b)).unwrap().0
	//{let e=nalgebra::linalg::SymmetricEigen::new(AtA).eigenvalues; assert_eq!(e.min(), e[8]);}
	//let Q = nalgebra::linalg::SymmetricEigen::new(AtA).eigenvectors;
	let H = Q.column(eigenvalues.iter().enumerate().min_by(|&(_, &a), &(_, b)| a.total_cmp(b)).unwrap().0);
	//let h = Q.column(8);
	let H: [f32; 9] = H.as_slice().try_into().unwrap();
	let H = H.map(|h| h / H[8]);
	//[[H[0],H[1],H[2]], [H[3],H[4],H[5]], [H[6],H[7],H[8]]]
	[[H[0],H[3],H[6]], [H[1],H[4],H[7]], [H[2],H[5],H[8]]]
	//( _invHnorm*_H0)*_Hnorm2;
	//_H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(2,2) );
}
