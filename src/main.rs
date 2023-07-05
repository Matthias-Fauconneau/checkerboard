#![feature(generators,iter_from_generator,array_methods,slice_flatten)]#![allow(non_camel_case_types,non_snake_case)]
use {vector::vec2, ::image::Image};
mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;
use ui::text::default_font;

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
        /*let center = P.into_iter().sum::<vec2>() / P.len() as f32;
        let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
        //[[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y]]
        [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]*/
        //[[scale.x, 0., 0.], [0., scale.y, 0.], [0., 0., 1.]]
        //let scale = num::sqrt(scale.x*scale.y); //println!("{scale}");
        //let scale = 1./40.;
        //[[scale, 0., 0.], [0., scale, 0.], [0., 0., 1.]]
        identity()
    });
    let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
    //let A = direct_linear_transform([P[0].map(|p| apply(M[0], p)), P[1].map(|p| apply(M[1], p))]);
    //let A = identity();
    //let A = mul(inverse(M[1]), mul(A, M[0]));
    /*println!("1 {:?}", P[1]);
    println!("0 {:?}", P[0]);
    println!("A {:?}", P[1].map(|p| apply(A, p)));*/

    if true {
        struct Plot<'t>(&'t [vec2]);
        impl ui::Widget for Plot<'_> { 
            #[ui::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { 
                for (i, &point) in self.0.iter().enumerate() {
                    use vector::{uint2, xy};
                    let p = target.size.y as f32*point/256.;
                    //let p = target.size.y as f32*(vec2::from(3./2.)+point)/3.;
                    let p = uint2::from(/*p.0.round()*/xy{x: p.x.round() as f32, y: p.y.round() as f32});
                    if p >= uint2::from(0) && p < target.size { target[p] = 0; }
                    let scale = num::Ratio{num: 30, div: ui::text::default_font()[0].height() as u32};
                    let label = format!("{}{}", i/4, i%4);
                    //let label = format!("{:.0} {:.0}", point.x, point.y);
                    ui::text(&label).paint(target, target.size, scale, vector::int2::from(p)/scale);
                }
            }
        }
        ui::run("Checkerboard", &mut Plot([P[1],P[0].map(|p| apply(A, p))].flatten())).unwrap();
        //ui::run("Checkerboard", &mut Plot([P[0].map(|p| apply(M[0], p)), P[1].map(|p| apply(A, apply(M[1], p)))].flatten())).unwrap();
    } else {       
        fn collect<I:Iterator, const N: usize>(mut iter: I) -> [I::Item; N] { let array = [(); N].map(|_| iter.next().unwrap()); assert!(iter.next().is_none()); array }
        let images : [_; 2] = collect(images.iter().zip([(identity(), P[0]), (A, P[1].map(|p| apply(inverse(A), p)))]).map(|(image, (A, P))| {
            let mut target = Image::zero(image.size);
            affine_blit(&mut target.as_mut(), image.as_ref(), A);
            let image = target;

            let max = *image.iter().max().unwrap();
            let mut target = image;
            //target.as_mut().map(|&v| (v as u16 * 0xFF / max as u16) as u8);
            Image::map(&mut target.as_mut(), |&v| (v as u16 * 0xFF / max as u16) as u8);

            for &p in &P {
                use vector::{uint2, xy};
                target[uint2::from(/*p.0.round()*/xy{x: p.x.round() as f32, y: p.y.round() as f32})] = 0xFF;
            }
            target
        }));

        struct View<'t, const N: usize> {
            images: [Image<&'t [u8]>; N],
            index: usize,
        }
        impl<const N: usize> ui::Widget for View<'_, N> { 
            #[ui::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { upscale(target, self.images[self.index].as_ref()) } 
            fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, _: &ui::Event) -> ui::Result<bool> {
                self.index = (self.index+1)%self.images.len();
                Ok(true)
            }
        }
        ui::run("Checkerboard", &mut View{images: images.each_ref().map(Image::as_ref), index: 0}).unwrap();
    }
}
