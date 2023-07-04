#![feature(generators,iter_from_generator,array_methods,slice_flatten)]#![allow(non_camel_case_types,non_snake_case)]
use {vector::vec2, ::image::Image};
mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;

fn main() {
    fn decode(path: impl AsRef<std::path::Path>) -> Image<Box<[u8]>> {
        let image = imagers::open(path).unwrap();
        Image::new(vector::xy{x: image.width(), y: image.height()}, image.into_bytes().into_boxed_slice())
    }
    let [mut nir, ir] = ["nir","ir"].map(|name| decode(format!("{name}.png")));
    pub fn invert(image: &mut Image<&mut [u8]>) { image.map(|&v| 0xFF-v) }
    invert(&mut nir.as_mut());
    let images = [nir, ir];
    let images = {use vector::xy; [images[0].slice(xy{x: 0, y: 0}, xy{x: images[0].size.y, y: images[0].size.y}), images[1].slice(xy{x: images[0].size.x-images[0].size.y, y: 0}, xy{x: images[0].size.y, y: images[0].size.y})]};
    let checkerboards = images.each_ref().map(|image| checkerboard(image.as_ref()));
    let mut P = checkerboards; P[1] = [P[1][0], P[1][3], P[1][1], P[1][2]];
    let M = P.map(|P| {
        let center = P.into_iter().sum::<vec2>() / P.len() as f32;
        let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
        [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
    });
    let A = direct_linear_transform([P[0].map(|p| apply(M[0], p)), P[1].map(|p| apply(M[1], p))]);
    /*println!("0 {:?}", P[0].map(|p| apply(M[0], p)));
    println!("1 {:?}", P[1].map(|p| apply(M[1], p)));
    println!("A {:?}", P[0].map(|p| apply(A, apply(M[0], p))));*/
    let A = mul(inverse(M[1]), mul(A, M[0]));
    /*println!("0 {:?}", P[0]);
    println!("1 {:?}", P[1]);
    println!("A {:?}", P[0].map(|p| apply(A, p)));*/

    let images : [_; 2] = std::array::from_fn(|i| {
        let mut target = Image::zero(images[i].size);
        affine_blit(&mut target.as_mut(), images[i].as_ref(), if i == 0 { identity() } else { A });
        let image = target;

        let max = *image.iter().max().unwrap();
        let mut target = image;
        target.as_mut().map(|&v| (v as u16 * 0xFF / max as u16) as u8);

        for &p in &if i == 0 { P[0] } else { P[1].map(|p| apply(inverse(A), p)) } {
            use vector::{uint2, xy};
            target[uint2::from(/*p.0.round()*/xy{x: p.x.round() as f32, y: p.y.round() as f32})] = 0xFF;
        }
        target
    });

    struct View<'t> {
        images: [Image<&'t [u8]>; 2],
        index: usize,
    }
    impl ui::Widget for View<'_> { 
        fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result { Ok(upscale(target, self.images[self.index].as_ref())) } 
        fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, _: &ui::Event) -> ui::Result<bool> {
            self.index = (self.index+1)%self.images.len();
            Ok(true)
        }
    }
    ui::run("Checkerboard", &mut View{images: images.each_ref().map(Image::as_ref), index: 0}).unwrap();
}
