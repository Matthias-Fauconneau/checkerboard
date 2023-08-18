#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows,slice_take,stdsimd)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}, ui::time};

pub trait Camera : Sized {
    fn new() -> Self;
    fn next(&mut self) -> Image<Box<[u16]>>;
    /*fn next_or_saved_or_start(camera: &mut Option<Self>, clone: &mut Option<Image<Box<[u16]>>>, name: &str, size: size) -> Image<Box<[u16]>> {
        if let Some(camera) = camera.as_mut() {let image = camera.next(); *clone = Some(image.clone()); image} 
        else if let Some(image) = raw(name,size) { image }
        else { *camera = Some(Self::new()); let image = camera.as_mut().unwrap().next(); *clone = Some(image.clone()); image}
    }*/
}
pub struct Unimplemented;
impl Camera for Unimplemented {
    fn new() -> Self { Self }
    fn next(&mut self) -> Image<Box<[u16]>> { unimplemented!() }
}

mod nir; use nir::NIR;
mod ir; use ir::IR; 

mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;
mod bt40; use bt40::*;

#[derive(Debug,Clone,Copy)] enum Mode { Place, Calibrate, Use, Replace }
struct Calibrated<T> {
    camera: T,
    calibration: [[vec2; 4]; 2],
}
impl<T:Camera> Calibrated<T> {
    /*fn P_identity(size: size) -> [vec2; 4] {
        let size = xy{x: nir_size.y*7/5/2, y: nir_size.y/2};
        let offset = (nir_size-size)/2;
        let side_length = size.x/7;
        [xy{x: side_length, y: side_length}, xy{x: size.x-side_length, y: side_length}, xy{x: size.x-side_length, y: size.y-side_length}, xy{x: side_length, y: size.y-side_length}].map(|p| vec2::from(offset+p))
    }*/
    fn new(name: &str/*, size: size*/) -> Self { 
        Self{
            camera: T::new(), 
            calibration: std::fs::read(name).map(|data| *bytemuck::from_bytes(&data)).unwrap_or(
                [/*Self::P_identity(size)*/[xy{x: 0., y: 0.}, xy{x: 1., y: 0.}, xy{x: 1., y: 1.}, xy{x: 0., y: 1.}]; 2]
            )
        }
    }
}
struct App {
    nir: Option<Calibrated<NIR>>,
    mode: Mode,
    last: Option<[vec2; 4]>,
    ir: Calibrated<IR>,
    _bt40: Option<BT40>,
    which: &'static str,
    debug: &'static str,
    //last_frame: [Option<Image<Box<[u16]>>>; 3],
}
impl App { 
    fn new() -> Self { Self{
        nir: None,//Calibrated::<NIR>::new("nir"/*,xy{x: 2048, y: 1152}*/),
        mode: Mode::Place,
        last: None, 
        ir: Calibrated::<IR>::new("ir"),
        _bt40: /*{let mut bt40=BT40::new(); bt40.enable(); bt40}*/None, 
        which: "ir", debug:"response", 
        //last_frame: [None, None, None], 
    }}
    fn calibration(&mut self) -> &mut [[vec2; 4]; 2] {
        match self.which {
            //"nir" => &mut self.nir.calibration,
            "ir" => &mut self.ir.calibration,
            _ => unimplemented!(),
        }
    }
}
impl ui::Widget for App {
    fn size(&mut self, _: size) -> size { xy{x:2592,y:1944} }
    fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
        if target.size == (xy{x: 960, y: 540}) { return Ok(()); } // WORKAROUND: winit first request paint at wrong scale factor
        
        /*let hololens = Camera::next_or_saved_or_start(&mut self.hololens,&mut self.last_frame[0], "hololens",xy{x:1280,y:720});
        let debug = if self.debug_which=="hololens" {self.debug} else{""};
        if debug=="original" { scale(target, hololens.as_ref()); return Ok(()) }
        let points = match checkerboard_direct_intersections(hololens.as_ref(), 64, debug) { Ok(points) => points, Err(image) => {scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = scale(target, hololens.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(()) }
        let P_hololens = long_edge_first(top_left_first(simplify(convex_hull(&points.into_iter().map(vec2::from).collect::<Box<_>>())).try_into().unwrap()));
        if debug=="quads" { let (_, scale, offset) = scale(target, hololens.as_ref()); for a in P_hololens { cross(target, scale, offset, a, 0xFF00FF); } return Ok(()) }*/

        /*let full_nir = self.nir.camera.next();
        //let nir = {let size = nir.size/128*128; nir.slice((nir.size-size)/2, size)}; // 2592x1944 => 2560x1920
        //let nir = {let size = xy{x: nir.size.x, y: nir.size.x/16*9}; nir.slice((nir.size-size)/2, size)};  // 16:9 => 2592x1458|2560x1440
        let nir = {let ref nir = full_nir; let size = xy{x: nir.size.x/128/16*16*128, y: nir.size.x/128/16*9*128}; nir.slice((nir.size-size)/2, size)};  // 16:9 => 2048x1152
        //let nir = {let x = nir.size.x/64*64; nir.slice(xy{x: (nir.size.x-x)/2, y: 0}, xy{x, y: nir.size.y})}; // Both dimensions need to be aligned because of transpose (FIXME: only align stride+height)*/

        let ir = self.ir.camera.next();
        
        //let scale = num::Ratio{num: 15, div: 16}; // 2048->1920 x 1152->1080
        //assert_eq!(scale*nir.size, target.size);
        //let scale = num::Ratio{num: 45, div: 8}; // 192->1080

        let (source, offset) = match self.which {
            //"nir" => (full_nir.as_ref(), (full_nir.size-nir.size)/2),
            "ir" => (ir.as_ref(), xy{x: 0, y: 0}),
            which => unreachable!("{which}"),
        };
        let scale = num::Ratio{num: target.size.y, div: source.size.y};
        
        let map = |P:[[vec2; 4]; 2]| -> mat3 {
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            mul(inverse(M[1]), mul(A, M[0]))
        };
        
        {let size=target.size; for x in 0..target.size.x { target[xy{x, y:0}] = 0xFFFFFF; target[xy{x, y: size.y-1}] = 0xFFFFFF; }}
        {let size=target.size; for y in 0..target.size.y { target[xy{x: 0, y}] = 0xFFFFFF; target[xy{x: size.x-1, y}] = 0xFFFFFF; }}
        if let Mode::Use = self.mode {
            let A = map(*self.calibration());
            let vector::MinMax{min, max} = vector::minmax(source.iter().copied()).unwrap(); // FIXME
            let scale = 1./f32::from(scale);
            for y in 0..target.size.y {
                for x in 0..target.size.x {
                    let p = scale*xy{x: x as f32, y: y as f32}; // ->source
                    let p = apply(A, p);
                    let p = vec2::from(offset)+p;
                    if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
                    let s = source[uint2::from(p)];
                    target[xy{x,y}] = u32::from(bgr8::from(((s-min)*0xFF/(max-min)) as u8)); // nearest: 2048x1152 -> 1920x1080
                }
            }
            return Ok(());
        }

        let debug = self.debug;
        let P = match self.which {
            /*"nir" => {
                if debug=="original" { image::scale(target, nir.as_ref()); return Ok(()) }
                let low = blur::<4, 4>(nir.as_ref()).unwrap(); // 9ms
                if debug=="blur" { image::scale(target, low.as_ref()); return Ok(()) }
                /*let high = self::normalize::<42>(low.as_ref(), 0x1000);
                if debug=="high" { scale(target, high.as_ref()); return Ok(()) }*/
                let high = low.as_ref();
                //let Some(P_nir) = checkerboard_quad_debug(high.as_ref(), true, 0/*Blur is enough*/ /*9*/, /*min_side: */64., debug, target)  else { return Ok(()) };
                //let high = nir.as_ref();
                let points = match checkerboard_direct_intersections::<32,4>(high.as_ref(), /*max_distance:*/64, debug) { Ok(points) => points, Err(image) => {image::scale(target, image.as_ref()); return Ok(()) }};
                if debug=="points" { let (_, scale, offset) = image::scale(target, high.as_ref()); for &a in &points { cross(target, scale, offset, a.into(), 0xFF0000); } return Ok(()) }
                let P = simplify(convex_hull(&points.iter().copied().map(vec2::from).collect::<Box<_>>())).try_into();
                let P = P.map(|P_nir| long_edge_first(top_left_first(P_nir)));
                let Ok(P) = P else { let (_, scale, offset) = image::scale(target, high.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00000); } return Ok(()); };
                if debug=="quads" { let (_, scale, offset) = image::scale(target, nir.as_ref()); for a in P { cross(target, scale, offset, a, 0xFF0000); } return Ok(()) }
                P
            }*/
            "ir" => {
                if debug=="original" { image::scale(target, ir.as_ref()); return Ok(()); }
                /*let Some(high) = checkerboard::normalize_slow/*::<16, 2>*/(ir.as_ref(), 16, 0x2000) else {image::scale(target, ir.as_ref()); return Ok(()) };
                if debug=="high" { image::scale(target, high.as_ref()); return Ok(()) };*///ir.as_ref();
                let points = match checkerboard_direct_intersections::<3, 2>(/*high*/ir.as_ref(), 32, debug) { Ok(points) => points, Err(image) => {image::scale(target, image.as_ref()); return Ok(()) }};
                if debug=="points" { let (_, scale, offset) = image::scale(target, ir.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(())}
                let P = fit_rectangle(&points);
                if debug=="quads" { let (_, scale, offset) = image::scale(target, ir.as_ref()); for a in P { cross(target, scale, offset, a, 0xFF00FF); } return Ok(())}
                P
            },
            _ => unreachable!()
        };

        self.last = Some(P);
        if let Mode::Replace = self.mode {
            self.calibration()[0] = P;
            self.mode = Mode::Calibrate; 
        }

        let offset = xy{x: 0, y: 0};
        let P = match self.mode {
            Mode::Place => P,
            Mode::Calibrate => {
                let scale = f32::from(scale);
                for (i,&p) in P.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }
                self.calibration()[0]
            },
            mode => unreachable!("{mode:?}"),
        };
        let scale = f32::from(scale);
        let [p00,p10,p11,p01] = P;
        target.fill(0);
        for i in 0..3 { for j in 0..5 {
            let color = if (i%2==1) ^ (j%2==1) { 0xFFFFFF } else { 0 };
            if color == 0 { continue; }
            let p = |di,dj| {
                let [i, j] = [i+di, j+dj];
                let [u, v] = [i as f32/3., j as f32/5.];
                let p0 = (1.-u)*p00+u*p01;
                let p1 = (1.-u)*p10+u*p11;
                let p = (1.-v)*p0+v*p1;
                vec2::from(offset)+scale*p
            };
            let P : [vec2; 4] = [p(0,0),p(0,1),p(1,1),p(1,0)];
            let vector::MinMax{min, max} = vector::minmax(P).unwrap();
            let [p00,p10,p11,p01] = P;
            for y in min.y as u32..=max.y as u32 { for x in min.x as u32..=max.x as u32 {
                let p = vec2::from(xy{x,y});
                if [p00,p10,p11,p01,p00].array_windows().all(|&[a,b]| vector::cross2(p-a,b-a)<0.) { if let Some(target) = target.get_mut(xy{x,y}) { *target = color; } }
            }}
        }}
        Ok(())
    }
    fn event(&mut self, _: vector::size, _context: &mut ui::EventContext, event: &ui::Event) -> ui::Result<bool> {
        if let &ui::Event::Key(key) = event {
            if ['⎙','\u{F70C}'].contains(&key) {
                println!("⎙");
                write("checkerboard.png", {let mut target = Image::uninitialized(xy{x: 2592, y:1944}); let size = target.size; self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap(); target}.as_ref());
                /*if self.last_frame.iter_mut().zip([/*"hololens",*/"nir","ir"]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| write_raw(name, image.as_ref())).count() == 0 {
                    //if self.hololens.is_none() { self.hololens = Some(Hololens::new()); }
                    //if self.nir.is_none() { self.nir = Some(Calibrated::<NIR>::new("nir")); }
                    //if self.ir.is_none() { self.ir = Some(Calibrated::<IR>::new("ir")); }
                }*/
            } else if key=='\n' {
                if let Some(last)= self.last {
                    self.calibration()[1] = last;
                    //context.set_title(&format!("{}", self.calibration.is_some()));
                    std::fs::write(self.which, bytemuck::bytes_of(&*self.calibration())).unwrap();
                    self.mode = Mode::Use; 
                }
            } else if key==' ' {
                self.mode = Mode::Replace;
            } else if key=='⌫' {
                self.mode = Mode::Place; 
            } else {
                fn starts_with<'t>(words: &[&'t str], key: char) -> Option<&'t str> { words.into_iter().find(|word| key==word.chars().next().unwrap()).copied() }
                if let Some(word) = starts_with(&["nir","ir"], key) { self.which = word }
                else if let Some(word) = starts_with(&["blur","contour","response","erode","high","low","max","original","z","points","quads","threshold"], key) { self.debug = word; }
                else { self.debug="";  }
                //context.set_title(&format!("{} {}", self.which, self.debug));
            }
            Ok(true)
        } else { Ok(/*self.hololens.is_some() ||*/ /*self.nir.is_some()*/ /*||self.ir.is_some()*/true) }
    }
}
#[cfg(debug_assertions)] const TITLE: &'static str = "#[cfg(debug_assertions)]";
#[cfg(not(debug_assertions))] const TITLE: &'static str = "Checkerboard";
fn main() -> ui::Result { ui::run(TITLE, &mut App::new()) }
