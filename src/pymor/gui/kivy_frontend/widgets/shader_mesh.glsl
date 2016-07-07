---VERTEX SHADER---
#ifdef GL_ES
    precision highp float;
#endif

attribute vec2  v_pos;
attribute float  v_color;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform vec2 scale;

varying float frag_color;

void main (void) {
    vec2 pos_moved = v_pos * scale;
    vec4 pos = modelview_mat * vec4(pos_moved,1.0,1.0);
    gl_Position = projection_mat * pos;
    frag_color = v_color;
}

---FRAGMENT SHADER---
#ifdef GL_ES
    precision highp float;
#endif

varying float frag_color;

uniform sampler2D tex;

vec3 getJetColor(float value) {
         float fourValue = 4.0 * value;
         float red   = min(fourValue - 1.5, -fourValue + 4.5);
         float green = min(fourValue - 0.5, -fourValue + 3.5);
         float blue  = min(fourValue + 0.5, -fourValue + 2.5);

         return clamp( vec3(red, green, blue), 0.0, 1.0 );
    }

vec3 getAntiJetColor(float value) {
         //R = -0.5*sin( L*(1.37*pi)+0.13*pi )+0.5;
         //G = -0.4*cos( L*(1.5*pi) )+0.4;
         //B =  0.3*sin( L*(2.11*pi) )+0.3;

         float pi = 3.1415926;

         float red   = -0.5*sin(value*1.37*pi+0.13*pi)+0.5;
         float green = -0.4*cos(value*1.50*pi)+0.4;
         float blue  = 0.3*sin(value*2.11*pi)+0.3;

         return clamp( vec3(red, green, blue), 0.0, 1.0 );
    }

void main (void){
    gl_FragColor = vec4(getAntiJetColor(frag_color), 1.0);
}