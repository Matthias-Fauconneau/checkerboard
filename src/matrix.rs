use std::array::from_fn as eval;
pub fn transpose<T: Copy, const M: usize, const N:usize>(a: [[T; N]; M]) -> [[T; M]; N] { eval(|i| eval(|j| a[j][i])) }

#[allow(non_camel_case_types)] type Matrix<const M: usize, const N:usize> = [[f32; N]; M];
#[allow(non_camel_case_types)] pub type mat3 = Matrix<3,3>;

//pub fn identity<const M: usize, const N:usize>() -> [[f32; M]; N] { eval(|i| eval(|j| if i==j {1.} else {0.})) }
pub fn mul<const M: usize, const N:usize, const P:usize>(a: Matrix<M,N>, b: Matrix<N,P>) -> Matrix<M,P> { eval(|i| eval(|j| (0..N).map(|k| a[i][k]*b[k][j]).sum())) }

use vector::vec2;
pub fn apply(M: mat3, v: vec2) -> vec2 { vec2::from(eval(|i| v.x*M[i][0]+v.y*M[i][1]+M[i][2]))/(v.x*M[2][0]+v.y*M[2][1]+M[2][2]) }

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
/*#[allow(non_camel_case_types)] pub type mat2x3 = Matrix<2,3>;
pub fn direct_linear_transform(P: [[vec2; 4]; 2]) -> mat2x3 {
	let eigen = nalgebra::SMatrix::<f32, 4, 6>::from_row_slice(&transpose(P).map(|[p,P]| [P.y*p.x, -P.x*p.x, P.y*p.y, -P.x*p.y, P.y, -P.x]).flatten()).svd().v_t;
	let H = eigen.eigenvectors.column(eigen.eigenvalues.argmin().0);
	let H: [_; 6] = H.as_slice().try_into().unwrap();
	//[[H[0],H[1]],[H[2],H[3]],[H[4],H[5]]]
	[[H[0],H[2],H[4]],[H[1],H[3],H[5]]]
}*/
pub fn _homography(P: [[vec2; 4]; 2]) -> mat3 {
	let mut AtA = nalgebra::SMatrix::<f32, 9, 9>::zeros();
	for [p,P] in transpose(P) {
		let x = [P.x,P.y,1., 0.,0.,0., -p.x*P.x, -p.x*P.y, -p.x];
		let y = [0.,0.,0., P.x,P.y,1., -p.y*P.x, -p.y*P.y, -p.y];
		for j in 0..9 { for i in j..9 { AtA[(i,j)] += x[i]*x[j] + y[i]*y[j]; } } // Lower triangle
	}
	let eigen = AtA.symmetric_eigen();
	let H = eigen.eigenvectors.column(eigen.eigenvalues.argmin().0);
	let H: [_; 9] = H.as_slice().try_into().unwrap();
	let H = H.map(|h| h / H[8]);
	[[H[0],H[1],H[2]], [H[3],H[4],H[5]], [H[6],H[7],H[8]]]
}

pub fn homography(P: [[vec2; 4]; 2]) -> mat3 {
	let M = P.map(|P| {
		let center = P.into_iter().sum::<vec2>() / P.len() as f32;
		let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
		[[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
	});
	let A = _homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
	mul(inverse(M[1]), mul(A, M[0]))
}