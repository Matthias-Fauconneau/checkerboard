#[allow(non_camel_case_types)] type Matrix<const M: usize, const N:usize> = [[f32; N]; M];
#[allow(non_camel_case_types)] pub type mat3 = Matrix<3,3>;

use std::array::from_fn as eval;
pub fn transpose<const M: usize, const N:usize>(a: Matrix<M,N>) -> Matrix<N,M> { eval(|i| eval(|j| a[j][i])) }
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
pub fn affine_transform([X,Y]: [[vec2; 4]; 2]) -> mat3 {
	let X = [X.map(|p| p.x), X.map(|p| p.y), X.map(|_| 1.)];
	let Y = [Y.map(|p| p.x), Y.map(|p| p.y), Y.map(|_| 1.)];
	mul(mul(X, transpose(Y)), inverse(mul(Y, transpose(Y))))
}
