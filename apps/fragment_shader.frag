//#version 330 core

#define PI 3.1415926535897932384626433832795
#define inf 9e100


out vec4 fragColor;

uniform vec2 resolution;
uniform int AA;
uniform vec2 mouse;
uniform float camera_distance;
uniform float time;
uniform float jitter_factor;
uniform int max_raymarching_steps; //64 by default
uniform float max_distance; //100 by default
uniform float step_size;

uniform float A, B, C, D, E;
uniform vec3 mov;

const vec3  kSunDir = normalize(vec3(-0.624695,0.168521,-0.624695));
vec3 _LightDir = kSunDir*100000.0; //we are assuming infinite light intensity
vec3 _CameraDir = vec3(0,5,5);

uniform float blinn_phong_alpha = 100;

uniform float ka = 0.05;
uniform float kd = 0.3;
uniform float ks = 1.0;

uniform float softshadows = 5.0;

uniform int shadows = 0;
uniform int clouds = 0;
uniform int gamma = 1;
uniform int lens_flare = 1;
uniform int hq_water = 1;

uniform int fbm_octaves = 8;
uniform int sha_octaves = 8;
uniform float sha_stepsize = 1.0;

uniform float fog = 1.0;
uniform float sun_dispersion = 1.0;

uniform float elevate = 2.0;
uniform float large = 0.5;
uniform float water_level = 100.0;

uniform float seed = 0.0; 




































// ------  VALUE NOISE BY INIGO QUIELEZ: http://www.iquilezles.org/www/articles/morenoise/morenoise.htm

float hash2( vec2 p )
{
    p  = 50.0*fract( p*0.3183099 );
    return fract( p.x*p.y*(p.x+p.y) );
}

float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 w = fract(x);
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);

    float a = hash2(p+vec2(0,0));
    float b = hash2(p+vec2(1,0));
    float c = hash2(p+vec2(0,1));
    float d = hash2(p+vec2(1,1));

    return -1.0+2.0*( a + (b-a)*u.x + (c-a)*u.y + (a - b - c + d)*u.x*u.y );
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



mat2 m2 = mat2(  0.80,  0.60, -0.60,  0.80 );
mat2 m2i = mat2( 0.80, -0.60, 0.60,  0.80 );

float fbm( in vec2 x, int octaves)
{
    float f = 1.9;
    float s = 0.50;
    float a = 0.0;
    float b = 0.5;
    for( int i=0; i< octaves; i++ )
    {
        float n = noise(x);
        a += b*n;
        b *= s;
        x = f*m2*x;
    }
	return a;
}

vec3 fbmd( in vec2 x , int octaves)
{
    float f = 1.9;
    float s = 0.50;
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


// ----- SKY FUNCTION BY INIGO QUIELEZ: https://www.shadertoy.com/view/4ttSWf

vec3 renderSky( in vec3 ro, in vec3 rd )
{
    // background sky     
    vec3 col = vec3(0);
    //col += vec3(1.0,0.1,0.1)*1.1 - rd.y*rd.y*0.5;

    float mysundot = clamp(dot(kSunDir,rd), 0.0, 1.0 );
    col += mix(vec3(1.0,0.1,0.1)*1.1, normalize(vec3(172, 133, 102)), rd.y*rd.y + 0.588*(1.0 - mysundot));

    //col += vec3(1.0,0.1,0.1)*1.1 - 2.0*rd.y*rd.y*0.5*vec3(1.0, 1.0, 0.0);


    //col += normalize(vec3(208,103,5))*1.1 - rd.y*rd.y*0.5;
    //col = mix( col, 0.85*vec3(0.7,0.75,0.85), pow( 1.0-max(rd.y,0.0), 4.0 ) );

    // clouds
    if (clouds > 0) {
        float t = (1000.0-ro.y)/rd.y;
        if( t>0.0 )
        {
            vec2 uv = (ro+t*rd).xz;
            float cl = fbm( uv*0.001 + vec2(-100.0, 20.0) , fbm_octaves);
            float dl = smoothstep(-0.2,0.6,cl);
            col = mix( col, vec3(1.0), 0.4*dl );
        }
    }

    // sun glare    
    float sundot = clamp(dot(kSunDir,rd), 0.0, 1.0 );
    //col += 0.25*vec3(1.0,0.7,0.4)*pow( sundot,5.0 );
    col += 20.0*normalize(vec3(250,206,150))*pow( sundot,2048.0);
    col += normalize(vec3(250,206,14))*pow( sundot,64.0 );
    col += normalize(vec3(244,162,5))*pow( sundot,5.0 );
    //col += 0.2*vec3(1.0,0.8,0.6)*pow( sundot,512.0 );

    //horizon
    col = mix( col, 0.5*normalize(vec3(118,34,8)), pow( 1.0-max(rd.y + 0.1,0.0), 10.0 ) );

    return col;
}



// ---- OTHER THINGS BY INIGO QUIELEZ

// return smoothstep and its derivative
vec2 smoothstepd( float a, float b, float x)
{
    if( x<a ) return vec2( 0.0, 0.0 );
    if( x>b ) return vec2( 1.0, 0.0 );
    float ir = 1.0/(b-a);
    x = (x-a)*ir;
    return vec2( x*x*(3.0-2.0*x), 6.0*x*(1.0-x)*ir );
}

float terrainMap(in vec2 p, int octaves) {
    float sca = 0.0010;
    float amp = 300.0;
    p *= sca;
    float e = fbm(p + vec2(1.0, -2.0)*(1.0 + seed), octaves);

    //e *= 2.0*E;



    e += elevate*exp(-large*(length(p)));

    //e *= (

    //canyons -- removed for now
    //float c = smoothstep(-0.08 , -0.01, e);
    //e = e + 0.15*c;

    e *= amp;
    return e;
}

vec4 terrainMapD( in vec2 p )
{
    float sca = 0.0010;
    float amp = 300.0;
    p *= sca;
    vec3 e = fbmd( p + vec2(1.0,-2.0)*(1.0 + seed) , fbm_octaves);

    //e.x *= 2.0*E;

    e.x += elevate*exp(-large*(length(p)));
    //e.y += elevate*exp(-large*(length(p)))*(large*2*p.x);
    //e.z += elevate*exp(-large*(length(p)))*(-large*2*p.y);

    //canyons -- removed for now
    //vec2 c = smoothstepd( -0.08, -0.01, e.x );
    //e.x = e.x + 0.15*c.x;
    //e.yz = e.yz + 0.15*c.y*e.yz;

    e.x *= amp;
    e.yz *= amp*sca;
    return vec4( e.x, normalize( vec3(-e.y,1.0,-e.z) ) );
}



// ----- FOG FUNCTION ADAPTED FROM INIGO QUIELEZ: http://www.iquilezles.org/www/articles/fog/fog.htm
vec3 applyFog(in vec3 rgb, in float dist, in vec3  rayDir, in vec3  sunDir )
{
    float fogAmount = 1.0 - exp( -(dist*dist/max_distance)*((fog*fog) / 200.0) );
    float sunAmount = clamp( dot( rayDir, sunDir ), 0.0, 1.0 );
    vec3 fogColor  = mix( 0.5*normalize(vec3(118,34,8)),
            vec3(1.0,0.9,0.1), // yellowish
            pow(sunAmount, (0.5/(sun_dispersion)) * 8.0 * ((dist) / 1000.0)) );
    return mix(rgb, fogColor, fogAmount );
}


// ----- LENS FLARE ADAPTED FROM NIMITZ: https://www.shadertoy.com/view/XtS3DD
float pent(in vec2 p){
    vec2 q = abs(p);
    return max(max(q.x*1.176-p.y*0.385, q.x*0.727+p.y), -p.y*1.237)*1.;
}

float circle(in vec2 p){
    return length(p);
}

vec3 flare(vec2 p, vec2 pos) 
{
	vec2 q = p-pos;
    vec2 pds = p*(length(p))*0.75;
	float a = atan(q.x,q.y);
    float rz = 0;

    vec2 p2 = mix(p,pds,-.5*D); //Reverse distort
    rz += max(0.01-pow(circle(p2 - 0.2*pos),1.7),.0)*3.0;
    rz += max(0.01-pow(pent(p2 + 0.4*pos),2.2),.0)*3.0;
    //rz += max(0.01-pow(pent(p2 + 0.2*pos),2.5),.0)*3.0;
    //rz += max(0.01-pow(pent(p2 - 0.1*pos),1.6),.0)*4.0;
    rz += max(0.01-pow(pent(-(p2 + 1.*pos)),2.5),.0)*5.0;
    rz += max(0.01-pow(pent(-(p2 - .5*pos)),2.),.0)*4.0;
    //rz += max(0.01-pow(pent(-(p2 + .7*pos)),2.),.0)*3.0;
    //rz += max(0.01-pow(pent(-(p2 + 1.4*pos)),1.9),.0)*3.0;
    //rz += max(0.01-pow(pent(-(p2 + 1.0*pos)),3.0),.0)*3.0;
    rz += max(0.01-pow(circle(-(p2 + 1.8*pos)),3.0),.0)*3.0;

    rz *= (1.0 - length(q));
    rz *= 5.0;

    return vec3(clamp(rz,0.,1.));
}


// ------WATER SHADER ADAPTED FROM FRANKENBURG: https://www.shadertoy.com/view/4sXGRM

float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 0.7; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 1.0; // change to adjust the small waves
vec3 watercolor  = vec3(0.2, 0.25, 0.3)*D;
// this calculates the water as a height of a given position
float water( vec2 p )
{
    vec2 shift2 = 0.0003*vec2( time*190.0*2.0, -time*130.0*2.0 );
    //float terrain = terrainMap(p - time*A, fbm_octaves);
    //shift2 *= clamp((water_level - terrain) / 500.0, 0.0, 1.0);

    float wave = 0.0;
    wave += sin(p.x * 0.021  + shift2.x) * 4.5;
    wave += sin(p.x * 0.0172 +p.y        * 0.010 + shift2.x * 1.121) * 4.0;
    wave -= sin(p.x * 0.00104+p.y        * 0.005 + shift2.x * 0.121) * 4.0;
    wave += sin(p.x * 0.02221+p.y        * 0.01233+shift2.x * 3.437) * 5.0;
    wave += sin(p.x * 0.03112+p.y        * 0.01122+shift2.x * 4.269) * 2.5 ;
    wave *= large_waveheight;
    wave -= fbm(p*0.004-shift2*.5 + vec2(1.0, -2.0), fbm_octaves)*small_waveheight*24.;
    // ...added by some distored random waves (which makes the water looks like water :)

    return water_level + wave;
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

float map(in vec3 p) {
    return 0.0;
}

vec3 normaleRubata(in vec3 p) {
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
        float terrain = terrainMap(p.xz - time*A, fbm_octaves);
        float d = p.y - terrain;


        if (p.y < water_level) {
            if (hq_water == 0) return vec4(0,0,1,0);
            t = ((water_level - ro.y) / rd.y); //adjust position to remove water artefacts
            p = ro + rd * t;
            vec3 col = vec3(0.0);

            wavegain *= E;

            // calculate water-mirror
            vec2 xdiff = vec2(0.1, 0.0)*wavegain*4.;
            vec2 ydiff = vec2(0.0, 0.1)*wavegain*4.;

            // get the reflected ray direction
            vec3 rd_bis = reflect(rd, normalize(vec3(water(p.xz-xdiff) - water(p.xz+xdiff), 1.0, water(p.xz-ydiff) - water(p.xz+ydiff))));
            float refl = 1.0-clamp(dot(rd_bis,vec3(0.0, 1.0, 0.0)),0.0,1.0);


            //reflection
            vec3 refl_col = renderSky(p, rd_bis);
            // raymarch to see if the sky is visible
            if (rd_bis.y > 0.0 && shadows > 0.0) {
                vec3 myro = p + rd_bis; //Start a bit higher than terrain. Helps when sha_octaves is low (< 5)
                vec3 myrd = rd_bis;
                float tt = 0.1;
                vec3 pp = p;
                for (int j = 0; j < max_raymarching_steps; j++) {
                    if (tt > max_distance)  break;
                    pp = myro + myrd * tt; 
                    terrain = terrainMap(pp.xz - time*A, sha_octaves);
                    d = pp.y - terrain;
                    if (d < 0.001*tt) {
                        refl_col = normalize(vec3(114, 53, 6))/2.0;
                        break;
                    }
                    tt += sha_stepsize*step_size * d;
                }
            }



            col = refl*0.5*refl_col;
            col += watercolor;

            col -= vec3(.3,.3,.3);

            terrain = terrainMap(p.xz - time*A, fbm_octaves);

            col += vec3(.15,.15,.15)*(clamp(20.0 / (water_level - terrain), 0.0, 1.0));
            col = applyFog(col, t, rd, kSunDir);

            float alpha = pow(t / max_distance, 3.0);
            if (alpha > 0.2) 
                col = mix(col, renderSky(ro, rd), alpha);


            return vec4(col, 1);

        }


        if (d < 0.002 * t) { 
            vec3 n = normaleRubata(p);
            vec3 col = vec3(ka);


            float diffuse_sun = clamp(kd * dot(kSunDir, n), 0.0, 1.0);
            //float specular = clamp(ks * pow(max(dot(n , normalize(kSunDir + normalize(_CameraDir - p))), 0.0), blinn_phong_alpha), 0.0, 1.0);
            float specular = 0.0;


            //shadows
            float shadow = 1.0;
            if (diffuse_sun > 0.01 && shadows > 0) {
                vec3 myro = p + kSunDir*15.0; //Start a bit higher than terrain. Helps when sha_octaves is low (< 5)
                vec3 myrd = kSunDir;
                float tt = 0.1;
                for (int j = 0; j < max_raymarching_steps; j++) {
                    if (tt > max_distance) break;
                    p = myro + myrd * tt; 
                    terrain = terrainMap(p.xz - time*A, sha_octaves);
                    d = p.y - terrain;
                    if (d < 0.001*tt) {
                        shadow = 0.0;
                        break;
                    }
                    shadow = min(shadow, softshadows*(d/tt));
                    tt += sha_stepsize*step_size * d;
                }
            }



            //col += diffuse_sun * vec3(1.64, 1.27, 0.99) * pow(vec3(shadow), vec3(1.0, 1.2, 1.5));
            //col += diffuse_sun * normalize(vec3(250,206,14))* pow(vec3(shadow), vec3(1.0, 1.2, 1.5));
            col += diffuse_sun * normalize(vec3(250,206,14))* pow(vec3(shadow), vec3(1.0, 1.2, 1.5));

            float indirect_sun = clamp(dot(n, normalize(kSunDir*vec3(-1.0, 0.0, -1.0))), 0.0, 1.0);

            col += indirect_sun *  vec3(0.40, 0.28, 0) * 0.14; // MULTIPLY PER AMBIENT OCCLUSION HERE

            float diffuse_sky = clamp(0.5 + 0.5*n.y, 0.0, 1.0);


            col += diffuse_sky * vec3(0.596, 0.182, 0.086) * 0.2;  // MULTIPLY PER AMBIENT OCCLUSION HERE

            col += specular;


            // far terrain disappears
            float alpha = pow(t / max_distance, 3.0);
            if (alpha > 0.2) 
                col = mix(col, renderSky(ro, rd), alpha - 0.2);
            
            col = applyFog(col, t, rd, kSunDir);

            return vec4(col, 1);
        }

        t += d*step_size;
    }

    vec3 sky = renderSky(ro, rd); 
    return vec4(sky.xyz, 1.0);
}


void main() {

    //camera movement
    if (mouse.y != 0.0) {
        _CameraDir.y =  max(5.0*(mouse.y / resolution.y - 0.5)*3.0,0.8); // to avoid going underwater
        _CameraDir.z = -5.0*cos(mouse.x / resolution.x * 2.0 * PI);
        _CameraDir.x = -5.0*sin(mouse.x / resolution.x * 2.0 * PI);
    }
    _CameraDir *= camera_distance;

    fragColor = vec4(0);

    for (int m = 0; m < AA; m++) {
        for (int n = 0; n < AA; n++) {
            vec2 uv = -1.0 + 2.0*(gl_FragCoord.xy/resolution) + vec2(n * (1.0/(resolution.x * AA)), m * (1.0/(resolution.y * AA)));
            uv.x *= resolution.x/resolution.y;

            vec3 rd = camera(_CameraDir, vec3(0.0, 200.0, 0.0))*normalize(vec3(uv, 2.0));

            fragColor += raymarch_terrain(_CameraDir, rd);
            //fragColor = vec4(fbm(uv*40.0, fbm_octaves));

            if (lens_flare > 0) {
                vec3 sunpos = inverse(camera(_CameraDir, vec3(0)))*kSunDir;
                if (sunpos.z > 0) fragColor += vec4(flare(uv, sunpos.xy), 1);
            }
        }
    }


    fragColor /= float(AA*AA);
    if (gamma > 0) fragColor = pow(fragColor, vec4(1.0 / 2.2));
}    
