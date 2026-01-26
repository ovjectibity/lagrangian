use std::collections::HashMap;
use std::ffi::{CStr, CString};

use crate::mujoco;
use crate::model_def::{Body, Geom, GeomType, WorldBody};

pub fn get_test_model() -> WorldBody {
    // A simple robot arm model
    WorldBody {
        bodies: vec![
            // Base of the arm
            Body {
                name: Some("base".to_string()),
                pos: Some(vec![0.0, 0.0, 0.25]),
                quat: None,
                euler: None,
                bodies: Some(vec![
                    // Upper arm
                    Body {
                        quat: None,
                        name: Some("upper_arm".to_string()),
                        pos: Some(vec![0.0, 0.0, 0.25]),
                        euler: Some(vec![0.0, -90.0, 0.0]),
                        bodies: Some(vec![
                            // Forearm
                            Body {
                                quat: None,
                                name: Some("forearm".to_string()),
                                pos: Some(vec![0.0, 0.5, 0.0]),
                                euler: Some(vec![0.0, 90.0, 0.0]),
                                bodies: Some(vec![
                                    // Hand
                                    Body {
                                        quat: None,
                                        name: Some("hand".to_string()),
                                        pos: Some(vec![0.0, 0.5, 0.0]),
                                        euler: Some(vec![0.0, -90.0, 0.0]),
                                        bodies: None,
                                        geoms: Some(vec![Geom {
                                            name: Some("hand_geom".to_string()),
                                            pos: Some(vec![0.0, 0.0, 0.0]),
                                            quat: None,
                                            euler: None,
                                            geom_t: Some(GeomType::Sphere),
                                            size: vec![0.15, 0.0, 0.0],
                                            mesh_obj: None,
                                        }]),
                                    },
                                ]),
                                geoms: Some(vec![Geom {
                                    name: Some("forearm_geom".to_string()),
                                    pos: Some(vec![0.0, 0.25, 0.0]),
                                    quat: None,
                                    euler: None,
                                    geom_t: Some(GeomType::Box),
                                    size: vec![0.05, 0.25, 0.05],
                                    mesh_obj: None,
                                }]),
                            },
                        ]),
                        geoms: Some(vec![Geom {
                            name: Some("upper_arm_geom".to_string()),
                            pos: Some(vec![0.0, 0.25, 0.0]),
                            quat: None,
                            euler: None,
                            geom_t: Some(GeomType::Box),
                            size: vec![0.05, 0.25, 0.05],
                            mesh_obj: None,
                        }]),
                    },
                ]),
                geoms: Some(vec![Geom {
                    name: Some("base_geom".to_string()),
                    pos: Some(vec![0.0, 0.0, 0.0]),
                    quat: None,
                    euler: None,
                    geom_t: Some(GeomType::Box),
                    size: vec![0.2, 0.2, 0.25],
                    mesh_obj: None,
                }]),
            },
        ],
    }
}

unsafe fn c_char_to_string(c_char_ptr: *const i8) -> 
Option<String> {
    unsafe {
        if c_char_ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(c_char_ptr).
            to_string_lossy().into_owned())
        }
    }
}

unsafe fn get_body_name(m: *const mujoco::mjModel, body_id: i32) -> 
Option<String> {
    unsafe {
        let name_adr = (*m).name_bodyadr.
            add(body_id as usize).read();
        if name_adr < 0 {
            return None;
        }
        let name_ptr = 
            (*m).names.add(name_adr as usize);
        c_char_to_string(name_ptr)
    }
}

unsafe fn get_geom_name(m: *const mujoco::mjModel, geom_id: i32) -> 
Option<String> {
    unsafe {
        let name_adr = (*m).name_geomadr.
            add(geom_id as usize).read();
        if name_adr < 0 {
            return None;
        }
        let name_ptr = 
            (*m).names.add(name_adr as usize);
        c_char_to_string(name_ptr)
    }
}

fn mujoco_geom_type_to_geom_type(mujoco_type: u32) -> 
Option<GeomType> {
    match mujoco_type {
        mujoco::mjtGeom__mjGEOM_PLANE => Some(GeomType::Plane),
        mujoco::mjtGeom__mjGEOM_SPHERE => Some(GeomType::Sphere),
        mujoco::mjtGeom__mjGEOM_CYLINDER => Some(GeomType::Cylinder),
        mujoco::mjtGeom__mjGEOM_CAPSULE => Some(GeomType::Capsule),
        mujoco::mjtGeom__mjGEOM_BOX => Some(GeomType::Box),
        mujoco::mjtGeom__mjGEOM_MESH => Some(GeomType::Mesh),
        _ => None,
    }
}

unsafe fn build_body_recursive(
    m: *const mujoco::mjModel,
    d: *const mujoco::mjData,
    body_id: i32,
    geoms_by_body: &HashMap<i32, Vec<Geom>>,
    children_by_body: &HashMap<i32, Vec<i32>>,
) -> Body {
    unsafe {
        let body_pos_ptr = (*d).xpos.add(body_id as usize * 3);
        let body_quat_ptr = (*d).xquat.add(body_id as usize * 4);

        let mut child_bodies = None;
        if let Some(child_ids) = children_by_body.get(&body_id) {
            let children: Vec<Body> = child_ids
                .iter()
                .map(|&child_id| {
                    build_body_recursive(
                        m, 
                        d, 
                        child_id, 
                        geoms_by_body, 
                        children_by_body)
                })
                .collect();
            if !children.is_empty() {
                child_bodies = Some(children);
            }
        }

        Body {
            name: get_body_name(m, body_id),
            pos: Some(
                std::slice::from_raw_parts(body_pos_ptr, 3)
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            ),
            quat: Some(
                std::slice::from_raw_parts(body_quat_ptr, 4)
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            ),
            euler: None, // MuJoCo provides quaternions
            bodies: child_bodies,
            geoms: geoms_by_body.get(&body_id).cloned(),
        }
    }
}

unsafe fn extract_mesh_data_as_obj(
    m: *const mujoco::mjModel, 
    mesh_id: i32) -> Option<String> {
    if mesh_id < 0 {
        return None;
    }

    let mesh_vertadr = (*m).mesh_vertadr.add(mesh_id as usize).read() as usize;
    let mesh_vertnum = (*m).mesh_vertnum.add(mesh_id as usize).read() as usize;
    let mesh_faceadr = (*m).mesh_faceadr.add(mesh_id as usize).read() as usize;
    let mesh_facenum = (*m).mesh_facenum.add(mesh_id as usize).read() as usize;

    let vertices = std::slice::from_raw_parts(
        (*m).mesh_vert.add(mesh_vertadr * 3),
        mesh_vertnum * 3,
    );
    let faces = std::slice::from_raw_parts(
        (*m).mesh_face.add(mesh_faceadr * 3),
        mesh_facenum * 3,
    );

    let mut obj_data = String::new();

    // Write vertices
    for i in 0..mesh_vertnum {
        let x = vertices[i * 3];
        let y = vertices[i * 3 + 1];
        let z = vertices[i * 3 + 2];
        obj_data.push_str(&format!("v {} {} {}\n", x, y, z));
    }

    // Write faces
    for i in 0..mesh_facenum {
        // OBJ format is 1-indexed
        let v1 = faces[i * 3] + 1;
        let v2 = faces[i * 3 + 1] + 1;
        let v3 = faces[i * 3 + 2] + 1;
        obj_data.push_str(&format!("f {} {} {}\n", v1, v2, v3));
    }

    Some(obj_data)
}


/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
pub unsafe fn create_model(
    m: *const mujoco::mjModel,
    d: *const mujoco::mjData,
) -> Option<WorldBody> {
    unsafe {
        let mut geoms_by_body: HashMap<i32, Vec<Geom>> = HashMap::new();
        for i in 0..(*m).ngeom {
            let geom_type_val = (*m).geom_type.add(i as usize).read();
            let geom_bodyid = (*m).geom_bodyid.add(i as usize).read();
            let geom_pos_ptr = (*m).geom_pos.add(i as usize * 3);
            let geom_quat_ptr = (*m).geom_quat.add(i as usize * 4);
            let geom_size_ptr = (*m).geom_size.add(i as usize * 3);
            let geom_dataid = (*m).geom_dataid.add(i as usize).read();

            let geom_t = mujoco_geom_type_to_geom_type(geom_type_val as u32);
            let mesh_obj = if let Some(GeomType::Mesh) = geom_t {
                extract_mesh_data_as_obj(m, geom_dataid)
            } else {
                None
            };

            let geom = Geom {
                name: get_geom_name(m, i),
                pos: Some(
                    std::slice::from_raw_parts(geom_pos_ptr, 3)
                        .iter()
                        .map(|&x| x as f32)
                        .collect(),
                ),
                quat: Some(
                    std::slice::from_raw_parts(geom_quat_ptr, 4)
                        .iter()
                        .map(|&x| x as f32)
                        .collect(),
                ),
                euler: None,
                geom_t,
                size: std::slice::from_raw_parts(geom_size_ptr, 3)
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
                mesh_obj,
            };
            geoms_by_body.entry(geom_bodyid).or_default().push(geom);
        }

        let mut children_by_body: HashMap<i32, Vec<i32>> = HashMap::new();
        for i in 1..(*m).nbody {
            // Skip world body at index 0
            let parent_id = (*m).body_parentid.add(i as usize).read();
            children_by_body.entry(parent_id).or_default().push(i);
        }

        let root_body_ids = 
            children_by_body.get(&0).cloned().unwrap_or_default();
        let bodies = root_body_ids
            .iter()
            .map(|&body_id| 
                build_body_recursive(
                    m, d, body_id, 
                    &geoms_by_body, 
                    &children_by_body))
            .collect();

        Some(WorldBody { bodies })
    }
}

unsafe extern "C" fn controller(m: *const mujoco::mjModel, d: *mut mujoco::mjData) {
    unsafe {
        //TODO: Hook the control to a model here
        let ctrl_d = (*m).nu;
        for _i in 0..ctrl_d {
            //Applying random actuation as sinusoidal:
            (*d).ctrl.add(0).write(100.0 * f64::sin((*d).time));
        }
    }
}

pub unsafe fn run_dynamics_step(
    m: *const mujoco::mjModel, d: *mut mujoco::mjData) {
    unsafe {
        println!("Running the simulation step for time {:?}",(*d).time);
        mujoco::mj_step(m, d);
    }
}

pub unsafe fn cleanup_model(m: *mut mujoco::mjModel, d: *mut mujoco::mjData) {
    unsafe {
        mujoco::mj_deleteData(d);
        mujoco::mj_deleteModel(m);
    }
}

pub unsafe fn load_model(
    model_name: String,
) -> Option<(*mut mujoco::mjModel, *mut mujoco::mjData)> {
    let mut error_buffer = [0i8; 1024];
    unsafe {
        let c_model_file_name = CString::new(model_name).
        expect("String contained null bytes");
        let model: *mut mujoco::mjModel = mujoco::mj_loadXML(
            c_model_file_name.as_ptr(),
            std::ptr::null_mut(),
            error_buffer.as_mut_ptr(),
            1024,
        );
        if model.is_null() {
            println!("Got null pointer while loading model {:?}", 
                c_char_to_string(error_buffer.as_ptr()));
        } else {
            println!("Model seems to have successfully loaded {}", (*model).nq);
            let data = mujoco::mj_makeData(model);
            if data.is_null() {
                println!("Got null point while loading data");
            }
            println!("Loading the controller for the model");
            mujoco::mjcb_control = Some(controller);
            println!("Number of actuator control dimension: {}", (*model).nu);
            return Some((model, data));
        }
    }
    None
}

pub unsafe fn load_test_model(file_name: String) -> Option<WorldBody> {
    unsafe {
        if let Some((m,d)) = load_model(file_name) {
            let c_m = create_model(m, d);
            return c_m;
        }
    }
    None
}

pub fn test_mujoco(model_name: String) {
    let mut error_buffer = [0i8; 1024];
    unsafe {
        // let model_file_name = "/Users/avi/Documents/manual/lagrangian/app/assets/slider_crank.xml";
        let c_model_file_name = CString::new(model_name).expect("String contained null bytes");
        let model: *mut mujoco::mjModel = mujoco::mj_loadXML(
            c_model_file_name.as_ptr(),
            std::ptr::null_mut(),
            error_buffer.as_mut_ptr(),
            1024,
        );
        if model.is_null() {
            println!("Got null point while loading model {:?}", error_buffer);
        } else {
            println!("Model seems to have successfully loaded {}", (*model).nq);
            let data = mujoco::mj_makeData(model);
            if data.is_null() {
                println!("Got null point while loading data");
            } else {
                println!("Loaded mjData as well for the model {}", (*data).nl);
                mujoco::mjcb_control = Some(controller);
                println!("NUmber of actuator control dimension: {}", (*model).nu);
                println!("Running the simulation now: ");
                mujoco::mj_step(model, data);
            }
            mujoco::mj_deleteData(data);
        }
        mujoco::mj_deleteModel(model);
    }
}
