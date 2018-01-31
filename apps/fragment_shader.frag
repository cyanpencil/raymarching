//#version 330 core

#define PI 3.1415926535897932384626433832795
#define inf 9e100


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


































//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
//

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
  return mod289(((x*34.0)+1.0)*x);
}

float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}








// ------  VALUE NOISE BY INIGO QUIELEZ: http://www.iquilezles.org/www/articles/morenoise/morenoise.htm

float hash2( vec2 p )
{
    p  = 50.0*fract( p*0.3183099 );
    return fract( p.x*p.y*(p.x+p.y) );
}


vec3 noised2( in vec2 x )
{
    vec2 p = floor(x);
    vec2 w = fract(x);
    
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec2 du = 30.0*w*w*(w*(w-2.0)+1.0);
    
    float a = hash2(p+vec2(0,0));
    float b = hash2(p+vec2(1,0));
    float c = hash2(p+vec2(0,1));
    float d = hash2(p+vec2(1,1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k4*u.x*u.y), 
                      2.0* du * vec2( k1 + k4*u.y,
                                      k2 + k4*u.x ) );
}


const mat2 m2 = mat2(  0.80,  0.60, -0.60,  0.80 );
const mat2 m2i = mat2( 0.80, -0.60, 0.60,  0.80 );

// returns 3D fbm and its 3 derivatives
vec3 fbmd_9( in vec2 x , int octaves)
{
    float f = 1.9;
    float s = 0.55;
    float a = 0.0;
    float b = 0.5;
    vec2  d = vec2(0.0);
    mat2  m = mat2(1.0,0.0,0.0,1.0);
    for( int i=0; i< octaves; i++ )
    {
        vec3 n = noised2(x);
        a += b*n.x;          // accumulate values		
        d += b*m*n.yz;       // accumulate derivatives
        b *= s;
        x = f*m2*x;
        m = f*m2i*m;
    }
	return vec3( a, d );
}










// --- Noise function by inigo quielez
vec2 hash( in vec2 x )  // replace this by something better
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    x = x*k + k.yx;
    return -1.0 + 2.0*fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}


// return gradient noise (in x) and its derivatives (in yz)
vec3 noised( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

#if 1
    // quintic interpolation
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
#else
    // cubic interpolation
    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
#endif    
    
    vec2 ga = hash( i + vec2(0.0,0.0) );
    vec2 gb = hash( i + vec2(1.0,0.0) );
    vec2 gc = hash( i + vec2(0.0,1.0) );
    vec2 gd = hash( i + vec2(1.0,1.0) );
    
    float va = dot( ga, f - vec2(0.0,0.0) );
    float vb = dot( gb, f - vec2(1.0,0.0) );
    float vc = dot( gc, f - vec2(0.0,1.0) );
    float vd = dot( gd, f - vec2(1.0,1.0) );

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                 du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}











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


// Inigo shapes
// All those shapes were taken from Inigo Quielez's blog:
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

//Sphere - signed - exact
float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}

//Box - unsigned - exact
float udBox( vec3 p, vec3 b ) {
  return length(max(abs(p)-b,0.0));
}

//Round Box - unsigned - exact
float udRoundBox( vec3 p, vec3 b, float r ) {
  return length(max(abs(p)-b,0.0))-r;
}

//Box - signed - exact
float sdBox( vec3 p, vec3 b ) {
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

//Torus - signed - exact
float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

//Cylinder - signed - exact
float sdCylinder( vec3 p, vec3 c ) {
  return length(p.xz-c.xy)-c.z;
}

//Cone - signed - exact
float sdCone( vec3 p, vec2 c ) {
    // c must be normalized
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}

//Plane - signed - exact
float sdPlane( vec3 p, vec4 n ) {
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}

//Hexagonal Prism - signed - exact
float sdHexPrism( vec3 p, vec2 h ) {
    vec3 q = abs(p);
    return max(q.z-h.y,max((q.x*0.866025+q.y*0.5),q.y)-h.x);
}

//Triangular Prism - signed - exact
float sdTriPrism( vec3 p, vec2 h ) {
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

//Capsule / Line - signed - exact
float sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

//Capped cylinder - signed - exact
float sdCappedCylinder( vec3 p, vec2 h ) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

//Capped Cone - signed - bound
float sdCappedCone( in vec3 p, in vec3 c ) {
    vec2 q = vec2( length(p.xz), p.y );
    vec2 v = vec2( c.z*c.y/c.x, -c.z );
    vec2 w = v - q;
    vec2 vv = vec2( dot(v,v), v.x*v.x );
    vec2 qv = vec2( dot(v,w), v.x*w.x );
    vec2 d = max(qv,0.0)*qv/vv;
    return sqrt( dot(w,w) - max(d.x,d.y) ) * sign(max(q.y*v.x-q.x*v.y,w.y));
}

//Ellipsoid - signed - bound
float sdEllipsoid( in vec3 p, in vec3 r ) {
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}


//Torus82 - signed
float sdTorus82( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length8(q)-t.y;
}

//Torus88 - signed
float sdTorus88( vec3 p, vec2 t ) {
  vec2 q = vec2(length8(p.xz)-t.x,p.y);
  return length8(q)-t.y;
}
















float mysdTorus88( vec3 p, vec2 t ) {
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


float sdCross( in vec3 p ) {
  float da = sdBox(p.xyz,vec3(inf,1.0,1.0));
  float db = sdBox(p.yzx,vec3(1.0,inf,1.0));
  float dc = sdBox(p.zxy,vec3(1.0,1.0,inf));
  return min(da,min(db,dc));
}













// return smoothstep and its derivative
vec2 smoothstepd( float a, float b, float x)
{
	if( x<a ) return vec2( 0.0, 0.0 );
	if( x>b ) return vec2( 1.0, 0.0 );
    float ir = 1.0/(b-a);
    x = (x-a)*ir;
    return vec2( x*x*(3.0-2.0*x), 6.0*x*(1.0-x)*ir );
}

vec4 terrainMapD( in vec2 p )
{
	const float sca = 0.0010;
    const float amp = 300.0;
    p *= sca;
    vec3 e = fbmd_9( p + vec2(1.0,-2.0) , 8);
    vec2 c = smoothstepd( -0.08, -0.01, e.x );
    e.x = e.x + 0.15*c.x;
    e.yz = e.yz + 0.15*c.y*e.yz;
    e.x *= amp;
    e.yz *= amp*sca;
    return vec4( e.x, normalize( vec3(-e.y,1.0,-e.z) ) );
}




float map(vec3 p) {
    //vec3 rp = p;
    //vec3 rp = twist(p);
    //vec3 rp = (inverse(rotateZ(time * 2)) * vec4(p, 1.0)).xyz;
    //return sdTorus_bulbus(rp, vec2(1, 0.2));

    //float d1 = sdSphere(mod(p + 5.0, 10.0) - 5.0, 1.2);
    //float d1 = sdSphere(vec3(p.x, p.y, mod(p.z, 10.0)), 1.2);
    //float d2 = sdSphere(p + mov/2.0, 11.5);
    //float d3 = sdTorus(twist((inverse(rotateY(time * 2)) * vec4(p + mov/10.0, 1.0)).xyz), vec2(1.3, 0.2));
    //vec3 rp = (inverse(rotateZ(time * 2)) * vec4(p, 1.0)).xyz;
    //float d4 = sdTorus(rp, vec2(2, 0.2));
    //return smin(d1, d2, 5.5);

    //return box(rp, 2.0 + pow(sin(p.x), C) + p.x/50.0);


    // --- menger sponge
    //
    //float d = sdBox(p,vec3(1.0));
    //float s = 1.0;
    //for( int m=0; m<3; m++ ) {
        //vec3 a = mod( p*s, 2.0 )-1.0;
        //s *= 3.0;
        //vec3 r = abs(1.0 - 3.0*abs(a));

        //float da = max(r.x,r.y);
        //float db = max(r.y,r.z);
        //float dc = max(r.z,r.x);
        //float c = (min(da,min(db,dc))-1.0)/s;

        //d = max(d,c);
    //}
    //return d;

    //vec3 noise_value = vec3(0);
    //noise_value += noised((p.xz - time) / (D * A)) * A; 
    //noise_value += noised((p.xz - time) / B) * B; 
    //noise_value += noised((p.xz - time) / C) * C; 
    //noise_value += noised((p.xz - time) / D) * D; 

    //vec3 noise_value = fbmd_9((p.xz - time) / 1000.0 + vec2(D, -2.0), 8) * 300.0;
    vec3 noise_value = terrainMapD(p.xz - time*A).xyz;

    float terrain = noise_value.x;

    if (terrain < p.y) {
        return (p.y - terrain);
    }
    else {
        return 0.0;
    }

}

vec3 normaleRubata(in vec3 p) {
    //vec3 noise_value = terrainMapD(p.xz - time*A).yzw;
    //return normalize(vec3(-noise_value.y, 1, -noise_value.z));
    return terrainMapD(p.xz - time*A).yzw;
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
            //vec3 n = calcNormal(p);
            vec3 n = normaleRubata(p);
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
    //fragColor = vec4(snoise(uv*40.0));
}    
