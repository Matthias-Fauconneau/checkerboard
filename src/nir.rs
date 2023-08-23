pub struct NIR{
    #[allow(unused)] camera: cameleon::Camera<cameleon::u3v::ControlHandle, cameleon::u3v::StreamHandle>,
    payload_receiver: cameleon::payload::PayloadReceiver
}
use image::{Image, xy};
impl super::Camera for NIR {
    fn new(serial_number: &str, gain: bool) -> Self {
        let mut camera = cameleon::u3v::enumerate_cameras().unwrap().into_iter().find(|c| c.info().serial_number == serial_number).unwrap();
        camera.open().unwrap();
        //camera.ctrl.sirm().unwrap().disable_stream(&mut camera.ctrl).unwrap();
        camera.load_context().unwrap();
        let mut params_ctxt = camera.params_ctxt().unwrap();
        //let acquisition_mode = params_ctxt.node("AcquisitionMode").unwrap().as_enumeration(&params_ctxt).unwrap();
        //acquisition_mode.set_entry_by_symbolic(&mut params_ctxt, "Continuous").unwrap();
        //acquisition_mode.set_entry_by_symbolic(&mut params_ctxt, "SingleFrame").unwrap();
        let acquisition_frame_rate = params_ctxt.node("AcquisitionFrameRate").unwrap().as_float(&params_ctxt).unwrap(); // 30fps
        let max = acquisition_frame_rate.max(&mut params_ctxt).unwrap();
        acquisition_frame_rate.set_value(&mut params_ctxt, max).unwrap(); // 28us
        let exposure_time = params_ctxt.node("ExposureTime").unwrap().as_float(&params_ctxt).unwrap();
        if gain {
            /*let max = exposure_time.max(&mut params_ctxt).unwrap();
            exposure_time.set_value(&mut params_ctxt, max).unwrap();*/
            let gain = params_ctxt.node("Gain").unwrap().as_float(&params_ctxt).unwrap();
            let max = gain.max(&mut params_ctxt).unwrap();
            gain.set_value(&mut params_ctxt, max).unwrap();
            /*let analog_gain = params_ctxt.node("AnalogGain").unwrap().as_float(&params_ctxt).unwrap();
            let max = analog_gain.max(&mut params_ctxt).unwrap();
            analog_gain.set_value(&mut params_ctxt, max).unwrap();
            let digital_gain = params_ctxt.node("DigitalGain").unwrap().as_float(&params_ctxt).unwrap();
            let max = digital_gain.max(&mut params_ctxt).unwrap();
            digital_gain.set_value(&mut params_ctxt, max).unwrap();*/
        } //else {
            exposure_time.set_value(&mut params_ctxt, 100.).unwrap(); // 15ms=66Hz
        //}
        Self{payload_receiver: camera.start_streaming(3).unwrap(), camera}
    }
    fn next(&mut self) -> Image<Box<[u16]>> {
        /*let payload = loop {
            let payload = self.payload_receiver.recv_blocking();
            match payload {
                Ok(payload) => { break payload; },
                Err(_) => { /*println!("{e:?}");*/ }
            };
        };*/
        let payload = self.payload_receiver.recv_blocking().unwrap();
        let &cameleon::payload::ImageInfo{width, height, ..} = payload.image_info().unwrap();
        let image = Image::new(xy{x: width as u32, y: height as u32}, payload.image().unwrap());
        let image = Image::from_iter(image.size, image.iter().map(|&u8| u8 as u16));
        self.payload_receiver.send_back(payload);
        image
    }
}
