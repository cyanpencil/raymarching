#version 330 core

#define PI 3.1415926535897932384626433832795


out vec4 fragColor;

uniform vec2 resolution;
uniform vec2 mouse;
uniform float camera_distance;
uniform float time;
uniform float jitter_factor;
uniform int max_raymarching_steps; //64 by default
uniform float max_distance; //100 by default
uniform float step_size;

uniform float A, B, C, D;
uniform vec3 mov;

vec3 _LightDir = vec3(0,500,0); //we are assuming infinite light intensity
vec3 _CameraDir = vec3(0,5,5);
float blinn_phong_alpha = 100;

uniform float ka = 0.05;
uniform float kd = 0.3;
uniform float ks = 1.0;











//#define MaxSteps 30
//#define MinimumDistance 0.0009
//#define normalDistance     0.0002

//#define Iterations 7
////#define PI 3.141592
//#define Scale 3.0
//#define FieldOfView 1.0
//#define Jitter 0.05
//#define FudgeFactor 0.7
//#define NonLinearPerspective 2.0
//#define DebugNonlinearPerspective false

//#define Ambient 0.32184
//#define Diffuse 0.5
//#define LightDir vec3(1.0)
//#define LightColor vec3(1.0,1.0,0.858824)
//#define LightDir2 vec3(1.0,-1.0,1.0)
//#define LightColor2 vec3(0.0,0.333333,1.0)
//#define Offset vec3(0.92858,0.92858,0.32858)

//vec2 rotate(vec2 v, float a) {
	//return vec2(cos(a)*v.x + sin(a)*v.y, -sin(a)*v.x + cos(a)*v.y);
//}

//// Two light sources. No specular 
//vec3 getLight(in vec3 color, in vec3 normal, in vec3 dir) {
	//vec3 lightDir = normalize(LightDir);
	//float diffuse = max(0.0,dot(-normal, lightDir)); // Lambertian
	
	//vec3 lightDir2 = normalize(LightDir2);
	//float diffuse2 = max(0.0,dot(-normal, lightDir2)); // Lambertian
	
	//return
	//(diffuse*Diffuse)*(LightColor*color) +
	//(diffuse2*Diffuse)*(LightColor2*color);
//}


//// DE: Infinitely tiled Menger IFS.
////
//// For more info on KIFS, see:
//// http://www.fractalforums.com/3d-fractal-generation/kaleidoscopic-%28escape-time-ifs%29/
//float DE(in vec3 z)
//{
	//// enable this to debug the non-linear perspective
	//if (DebugNonlinearPerspective) {
		//z = fract(z);
		//float d=length(z.xy-vec2(0.5));
		//d = min(d, length(z.xz-vec2(0.5)));
		//d = min(d, length(z.yz-vec2(0.5)));
		//return d-0.01;
	//}
	//// Folding 'tiling' of 3D space;
	//z  = abs(1.0-mod(z,2.0));

	//float d = 1000.0;
	//for (int n = 0; n < Iterations; n++) {
		//z.xy = rotate(z.xy,4.0+2.0*cos( time/8.0));		
		//z = abs(z);
		//if (z.x<z.y){ z.xy = z.yx;}
		//if (z.x< z.z){ z.xz = z.zx;}
		//if (z.y<z.z){ z.yz = z.zy;}
		//z = Scale*z-Offset*(Scale-1.0);
		//if( z.z<-0.5*Offset.z*(Scale-1.0))  z.z+=Offset.z*(Scale-1.0);
		//d = min(d, length(z) * pow(Scale, float(-n)-1.0));
	//}
	
	//return d-0.001;
//}

//// Finite difference normal
//vec3 getNormal(in vec3 pos) {
	//vec3 e = vec3(0.0,normalDistance,0.0);
	
	//return normalize(vec3(
			//DE(pos+e.yxx)-DE(pos-e.yxx),
			//DE(pos+e.xyx)-DE(pos-e.xyx),
			//DE(pos+e.xxy)-DE(pos-e.xxy)
			//)
		//);
//}

//// Solid color 
//vec3 getColor(vec3 normal, vec3 pos) {
	//return vec3(1.0);
//}


//// Pseudo-random number
//// From: lumina.sourceforge.net/Tutorials/Noise.html
//float rand(vec2 co){
	//return fract(cos(dot(co,vec2(4.898,7.23))) * 23421.631);
//}

//vec4 rayMarch(in vec3 from, in vec3 dir, in vec2 fragCoord) {
	//// Add some noise to prevent banding
	//float totalDistance = Jitter*rand(fragCoord.xy+vec2(time));
	//vec3 dir2 = dir;
	//float distance;
	//int steps = 0;
	//vec3 pos;
	//for (int i=0; i < max_raymarching_steps; i++) {
		//// Non-linear perspective applied here.
		//dir.zy = rotate(dir2.zy,totalDistance*cos( time/4.0)*NonLinearPerspective);
		
		//pos = from + totalDistance * dir;
		//distance = DE(pos)*FudgeFactor;
		//totalDistance += distance*step_size;
		//if (distance < MinimumDistance) break;
		//steps = i;
	//}
	
	//// 'AO' is based on number of steps.
	//// Try to smooth the count, to combat banding.
	//float smoothStep =   float(steps) + distance/MinimumDistance;
	//float ao = 1.1-smoothStep/float(MaxSteps);
	
	//// Since our distance field is not signed,
	//// backstep when calc'ing normal
	//vec3 normal = getNormal(pos-dir*normalDistance*3.0);
	
	//vec3 color = getColor(normal, pos);
	//vec3 light = getLight(color, normal, dir);
	//color = (color*Ambient+light)*ao;
	//return vec4(color,1.0);
//}

//void main()
//{
	//// Camera position (eye), and camera target
	//vec3 camPos = 0.5*time*vec3(1.0,0.0,0.0);
	////vec3 target = camPos + vec3(1.0,0.0*cos(time),0.0*sin(0.4*time));
    //vec3 target = camPos + vec3(1.0,0.0,0.0);
	//vec3 camUp  = vec3(0.0,1.0,0.0);
	
	//// Calculate orthonormal camera reference system
	//vec3 camDir   = normalize(target-camPos); // direction for center ray
	////camUp = normalize(camUp-dot(camDir,camUp)*camDir); // orthogonalize
	//vec3 camRight = normalize(cross(camDir,camUp));
	
	//vec2 coord =-1.0+2.0*gl_FragCoord.xy/resolution.xy;
	//coord.x *= resolution.x/resolution.y;
	
	//// Get direction for this pixel
	//vec3 rayDir = normalize(camDir + (coord.x*camRight + coord.y*camUp)*FieldOfView);
	
	//fragColor = rayMarch(camPos, rayDir, gl_FragCoord.xy );
//}

//void main() {
    //vec2 uv = -1.0 + 2.0*(gl_FragCoord.xy/resolution);
    //uv.x *= resolution.x/resolution.y;

    //vec3 col = vec3(0);

    //float atime = time*0.3;
    ////vec3 ro = 2.5*vec3(cos(atime), 0, -sin(atime));
    ////camera movement
    //if (mouse.y != 0.0) {
        //_CameraDir.y = 5.0*(mouse.y / resolution.y - 0.5)*3.0;
        //_CameraDir.z =-5.0*cos(mouse.x / resolution.x * 2.0 * PI);
        //_CameraDir.x = 5.0*sin(mouse.x / resolution.x * 2.0 * PI);
    //}
    //_CameraDir *= camera_distance;
    //vec3 rd = camera(_CameraDir, vec3(0))*normalize(vec3(uv, 2.0));

    //fragColor = raymarch(_CameraDir, rd);
//}    








































mat4 rotateY(float theta) {
    return mat4(
        vec4(cos(theta), 0, sin(theta), 0),
        vec4(0, 1, 0, 0),
        vec4(-sin(theta), 0, cos(theta), 0),
        vec4(0, 0, 0, 1)
    );
}

mat4 rotateX(float theta) {
    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, cos(theta), -sin(theta), 0),
        vec4(0, sin(theta), cos(theta), 0),
        vec4(0, 0, 0, 1)
    );
}

mat4 rotateZ(float theta) {
    return mat4(
        vec4(cos(theta), -sin(theta), 0, 0),
        vec4(sin(theta), cos(theta), 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    );
}

// Pseudo-random number
// From: lumina.sourceforge.net/Tutorials/Noise.html
float rand(vec2 co){
    return fract(cos(dot(co,vec2(4.898,7.23))) * 23421.631);
}

// Adapted from: https://stackoverflow.com/questions/26070410/robust-atany-x-on-glsl-for-converting-xy-coordinate-to-angle
float atan2(in float y, in float x) {
    return x == 0.0 ? sign(y)*PI/2 : atan(y, x);
}


float length8(vec2 p) {
    vec2 r = p*p*p*p;
    return pow(dot(r,r), (1.0/8.0));
}

// --------------------------------------------------------------------
// ------------------------- TRANSFORMATIONS --------------------------
// --------------------------------------------------------------------

vec3 jumping_twist2( vec3 p ) {
    float c = cos(5.0*p.y);
    float s = sin(5.0*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz *sin(time),p.y);
    return q;
}

vec3 jumping_twist( vec3 p ) {
    float c = cos(5.0*p.y);
    float s = sin(5.0*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y*pow(sin(time)*1.5, 4.0));
    return q;
}

vec3 twist( vec3 p ) {
    float c = cos(5.0*p.y);
    float s = sin(5.0*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return q;
}


// --------------------------------------------------------------------
// ------------------------- OPERATIONS -------------------------------
// --------------------------------------------------------------------

//Polynomial smooth minimum from: http://iquilezles.org/www/articles/smin/smin.htm
float smin(float a, float b, float k ) {
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float opSubtract( float d1, float d2 ) {
    return max(-d1,d2);
}

float opUnion(float d1, float d2) {
    return min(d1, d2);
}


// --------------------------------------------------------------------
// ---------------------- DISTANCE FUNCTIONS --------------------------
// --------------------------------------------------------------------

float sdTorus88( vec3 p, vec2 t ) {
  vec2 q = vec2(length8(p.xz)-t.x,p.y);
  float angle = atan2(p.z , p.x)/2 + time*D; //atan divided by 2 to  have only one bulb
  return length8(q) - (t.y + exp(A*(pow(cos(angle), C)))/B);
}


float sdTorus_bulbus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    //return length(q) - (t.y + (pow(max(0.0, sin(atan2(p.z , p.x) + time*2)), 10)/3));


    float angle = atan2(p.z , p.x)/2 + time*D; //atan divided by 2 to  have only one bulb
    //return length(q) - (t.y + exp(-100*(pow(cos(angle), 2)))/5);
    return length(q) - (t.y + exp(A*(pow(cos(angle), C)))/B);

    //In latex: \sqrt{(x - T_x)^2 + (y)^2} - T_y + e^{-100*\cos(atan2(y, x)/2 + t)^2 / 5}
}

// Adapted from: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - (t.y );
}

float sphere(vec3 p, float r) {
    return length(p) - r;
}


float stupida(vec3 p) {
    //return length(p) - 2.0 + sin(p.x * 10.0) / 50.0 + sin(p.y * 10.0) / 50.0;
    //return length(p) - 2.0 + pow(sin(atan2(p.y, p.x)*20.0) / 5.0, 2) + pow(sin(atan2(p.z, p.x)*20.0) / 5.0, 2);

    if (p.z <= -1.0 || p.z >= 1.0) return length(p) - 1.0;
    if (p.x <= -1.0 || p.x >= 1.0) return length(p) - 1.0;
    if (p.y <= -1.0 || p.y >= 1.0) return length(p) - 1.0;
    float phi = acos(p.z);
    float theta = atan2(p.y, p.x);

    //return length(p) - 1.0 + sin(theta*B * sin(phi)) / 10.0 * sin(phi*20.0) / 10.0;

    //prendo il punto 0,0,1
    float theta_p = acos(p.y);
    float theta_q = acos(0.0);
    float phi_p = atan2(p.z, p.x);
    float phi_q = atan2(0, 1.0);
    float spheric_dist = acos(cos(phi_q - phi_p)*cos(theta_p)*cos(theta_q) + sin(theta_p)*sin(theta_q));

    return length(p) - 1.0 + cos(spheric_dist * 20.0 + time) / 90.0;

    //return length(p) - 1.0 + sin(theta*20.0 / sin(phi)) / 50.0;


    //return length(p) - 1.0 + pow(sin(atan2(p.y, p.x)*20.0) / 5.0, 2) * sin(acos(p.z)*20.0) / 20.0;
}


float box (vec3 p, float l) {
    return max(max(abs(p.x) , abs(p.y)) , abs(p.z)) - l;
}




float map(vec3 p) {
    //vec3 rp = p;
    //vec3 rp = twist(p);
    vec3 rp = (inverse(rotateZ(time * 2)) * vec4(p, 1.0)).xyz;
    //return sdTorus_bulbus(rp, vec2(1, 0.2));

    float d1 = sphere(p, 1.2);
    float d2 = sphere(p + mov/4.0, 1.5);
    //float d3 = sdTorus(twist((inverse(rotateY(time * 2)) * vec4(p + mov/10.0, 1.0)).xyz), vec2(1.3, 0.2));
    //vec3 rp = (inverse(rotateZ(time * 2)) * vec4(p, 1.0)).xyz;
    //float d4 = sdTorus(rp, vec2(2, 0.2));
    return smin(d1, d2, 0.5);

    //return box(rp, 2.0 + pow(sin(p.x), C) + p.x/50.0);

}

//Adapted from: http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/#rotation-and-translation
vec3 calcNormal(in vec3 p) {
    float eps = 0.0001;
    return normalize(vec3(
        map(vec3(p.x + eps, p.y, p.z)) - map(vec3(p.x - eps, p.y, p.z)),
        map(vec3(p.x, p.y + eps, p.z)) - map(vec3(p.x, p.y - eps, p.z)),
        map(vec3(p.x, p.y, p.z + eps)) - map(vec3(p.x, p.y, p.z - eps))
    ));
}

//Adapted from: https://github.com/zackpudil/raymarcher/blob/master/src/shaders/scenes/earf_day.frag
mat3 camera(vec3 e, vec3 l) {
    vec3 f = normalize(l - e);
    vec3 r = cross(vec3(0, 1, 0), f);
    vec3 u = cross(f, r);
    
    return mat3(r, u, f);
}









//Adapted from: http://flafla2.github.io/2016/10/01/raymarching.html
vec4 raymarch(vec3 ro, vec3 rd) {
    vec4 ret = vec4(0,0,0,0);

    float initial_jitter = 0.0; //camera will jitter when too close to an object
    if (jitter_factor > 0.01) {
        initial_jitter = jitter_factor*rand(gl_FragCoord.xy + vec2(time));
    }

    float t = initial_jitter; // current distance traveled along ray
    for (int i = 0; i < max_raymarching_steps; ++i) {
        if (t > max_distance) break;
        vec3 p = ro + rd * t; //point hit on the surface
        float d = map(p);   

        if (d < 0.0001) {
            vec3 n = calcNormal(p);
            float l = ka;
            l += max(0.0, kd * dot(normalize(_LightDir - p), n));
            vec3 h = normalize(normalize(_LightDir - p) + normalize(_CameraDir - p));
            l += ks * pow(max(dot(n , h), 0.0), blinn_phong_alpha);
            ret = vec4(vec3(l,l,l), 1);
            break;
        }

        t += d*step_size;
    }
    return ret;
}



void main() {
    vec2 uv = -1.0 + 2.0*(gl_FragCoord.xy/resolution);
    uv.x *= resolution.x/resolution.y;

    vec3 col = vec3(0);

    float atime = time*0.3;
    //vec3 ro = 2.5*vec3(cos(atime), 0, -sin(atime));
    //camera movement
    if (mouse.y != 0.0) {
        _CameraDir.y = 5.0*(mouse.y / resolution.y - 0.5)*3.0;
        _CameraDir.z =-5.0*cos(mouse.x / resolution.x * 2.0 * PI);
        _CameraDir.x = 5.0*sin(mouse.x / resolution.x * 2.0 * PI);
    }
    _CameraDir *= camera_distance;
    vec3 rd = camera(_CameraDir, vec3(0))*normalize(vec3(uv, 2.0));

    fragColor = raymarch(_CameraDir, rd);
}    
