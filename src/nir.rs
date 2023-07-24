#[cfg(not(feature="u3v"))] pub struct NIR;
#[cfg(feature="u3v")] pub struct NIR{
    #[allow(dead_code)] camera: cameleon::Camera<cameleon::u3v::ControlHandle, cameleon::u3v::StreamHandle>,
    payload_receiver: cameleon::payload::PayloadReceiver
}
use image::{Image, xy};
#[cfg(not(feature="u3v"))] impl super::Camera for NIR {
    fn new() -> Self { Self }
    fn next(&mut self) -> Image<Box<[u16]>> { panic!("!u3v") }
}
#[cfg(feature="u3v")] impl super::Camera for NIR {
    fn new() -> Self {
        let mut cameras = cameleon::u3v::enumerate_cameras().unwrap();
        //for camera in &cameras { println!("{:?}", camera.info()); }
        let mut camera = cameras.remove(0); // find(|c| c.info().contains("U3-368xXLE-NIR")).unwrap()
        camera.open().unwrap();
        camera.load_context().unwrap();
        let mut params_ctxt = camera.params_ctxt().unwrap();
        let acquisition_frame_rate = params_ctxt.node("AcquisitionFrameRate").unwrap().as_float(&params_ctxt).unwrap(); // 30fps
        let max = acquisition_frame_rate.max(&mut params_ctxt).unwrap();
        acquisition_frame_rate.set_value(&mut params_ctxt, max).unwrap(); // 28us
        let exposure_time = params_ctxt.node("ExposureTime").unwrap().as_float(&params_ctxt).unwrap();
        exposure_time.set_value(&mut params_ctxt, 100.).unwrap(); // 15ms=66Hz
        Self{payload_receiver: camera.start_streaming(3).unwrap(), camera}
    }
    fn next(&mut self) -> Image<Box<[u16]>> {
        let payload = self.payload_receiver.recv_blocking().unwrap();
        let &cameleon::payload::ImageInfo{width, height, ..} = payload.image_info().unwrap();
        let image = Image::new(xy{x: width as u32, y: height as u32}, payload.image().unwrap());
        let image = Image::from_iter(image.size, image.iter().map(|&u8| u8 as u16));
        self.payload_receiver.send_back(payload);
        image
    }
}
