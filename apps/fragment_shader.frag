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

vec3 _LightDir = vec3(-0.624695,0.468521,-0.624695)*100000.0; //we are assuming infinite light intensity
vec3 _CameraDir = vec3(0,5,5);

uniform float blinn_phong_alpha = 100;

uniform float ka = 0.05;
uniform float kd = 0.3;
uniform float ks = 1.0;

uniform int shadows = 0;
uniform int clouds = 0;




































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


vec3 fbmd_9( in vec2 x , int octaves)
{
    float f = 1.9;
    float s = 0.55;
    float a = 0.0;
    float b = 0.5;
    vec2  d = vec2(0.0);
    mat2  m = mat2(1.0,0.0,0.0,1.0);
    mat2 m2 = mat2(  0.80,  0.60, -0.60,  0.80 );
    mat2 m2i = mat2( 0.80, -0.60, 0.60,  0.80 );
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


// ----- SKY FUNCTION BY INIGO QUIELEZ: https://www.shadertoy.com/view/4ttSWf

const vec3  kSunDir = vec3(-0.624695,0.468521,-0.624695);

vec3 renderSky( in vec3 ro, in vec3 rd )
{
    // background sky     
    vec3 col = 0.9*vec3(0.4,0.65,1.0) - rd.y*vec3(0.4,0.36,0.4);

    // clouds
    if (clouds > 0) {
        float t = (1000.0-ro.y)/rd.y;
        if( t>0.0 )
        {
            vec2 uv = (ro+t*rd).xz;
            float cl = fbmd_9( uv*0.002 , 8).x;
            float dl = smoothstep(-0.2,0.6,cl);
            col = mix( col, vec3(1.0), 0.4*dl );
        }
    }
    
    // sun glare    
    float sun = clamp( dot(kSunDir,rd), 0.0, 1.0 );
    col += 0.6*vec3(1.0,0.6,0.3)*pow( sun, 32.0 );
    
    return col;
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







vec4 raymarch_terrain(vec3 ro, vec3 rd) {
    vec4 ret = vec4(0,0,0,0);

    float initial_jitter = 0.0; //camera will jitter when too close to an object
    if (jitter_factor > 0.01) {
        initial_jitter = jitter_factor*rand(gl_FragCoord.xy + vec2(time));
    }

    float t = initial_jitter; // current distance traveled along ray
    for (int i = 0; i < max_raymarching_steps; ++i) {
        if (t > max_distance) break;
        vec3 p = ro + rd * t; //point hit on the surface

        vec4 noise_value = terrainMapD(p.xz - time*A);
        float terrain = noise_value.x;
        float d = p.y - terrain;
        if (d < 0.001 * t) { 
            vec3 n = noise_value.yzw;
            float l = ka;
            bool sunlight = true;




            l += clamp(kd * dot(kSunDir, n), 0.0, 1.0);
            vec3 h = normalize(kSunDir + normalize(_CameraDir - p));
            l += ks * pow(max(dot(n , h), 0.0), blinn_phong_alpha);
            ret = vec4(vec3(l,l,l), 1);



            //finire ombre
            if (shadows > 0) {
                vec3 myro = p;
                vec3 myrd = kSunDir;
                float tt = 0.01;
                for (int j = 0; j < 30; j++) {
                    p = myro + myrd * tt; 
                    noise_value = terrainMapD(p.xz - time*A);
                    terrain = noise_value.x;
                    d = p.y - terrain;
                    if (d < 0.001) {
                        return vec4(0,0,0,0);
                        break;
                    }
                    tt += step_size * d;
                }
            }
            //finire ombre


            return ret;
        }

        t += d*step_size;
    }

    vec3 sky = renderSky(ro, rd); 
    return vec4(sky.xyz, 1.0);
}




//Adapted from: http://flafla2.github.io/2016/10/01/raymarching.html
//vec4 raymarch(vec3 ro, vec3 rd) {
    //vec4 ret = vec4(0,0,0,0);

    //float initial_jitter = 0.0; //camera will jitter when too close to an object
    //if (jitter_factor > 0.01) {
        //initial_jitter = jitter_factor*rand(gl_FragCoord.xy + vec2(time));
    //}

    //float t = initial_jitter; // current distance traveled along ray
    //for (int i = 0; i < max_raymarching_steps; ++i) {
        //if (t > max_distance) break;
        //vec3 p = ro + rd * t; //point hit on the surface
        //float d = map(p);   

        //if (d < 0.0001) {
            //vec3 n = calcNormal(p);
            //float l = ka;
            //l += max(0.0, kd * dot(normalize(_LightDir - p), n));
            //vec3 h = normalize(normalize(_LightDir - p) + normalize(_CameraDir - p));
            //l += ks * pow(max(dot(n , h), 0.0), blinn_phong_alpha);
            //ret = vec4(vec3(l,l,l), 1);
            //break;
        //}

        //t += d*step_size;
    //}
    //return ret;
//}



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

    //fragColor = raymarch(_CameraDir, rd);
    fragColor = raymarch_terrain(_CameraDir, rd);
    //fragColor = vec4(snoise(uv*40.0));
}    
