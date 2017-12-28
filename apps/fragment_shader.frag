#version 330 core

#define PI 3.1415926535897932384626433832795


out vec4 fragColor;

uniform vec2 resolution;
uniform vec2 mouse;
uniform float camera_distance;
uniform float time;
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






float map(vec3 p) {
    vec3 rp = p;
    //vec3 rp = twist(p);
    //vec3 rp = (inverse(rotateZ(time * 2)) * vec4(p, 1.0)).xyz;
    //return sdTorus(rp, vec2(1, 0.2));
    float d1 = sphere(p, 1.0);
    float d2 = sphere(p + mov/4.0, 1.5);
    return smin(d1, d2, 0.5);
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

    float t = 0; // current distance traveled along ray
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
