#include "Reshade.fxh"

//
// As per reshade documentation
void EdgePostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
	if (id == 2)
		texcoord.x = 2.0;
	else
		texcoord.x = 0.0;

	if (id == 1)
		texcoord.y = 2.0;
	else
		texcoord.y = 0.0;

	position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

uniform float _DepthThreshold <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 5.0f;
    ui_type             = "slider";
    ui_label            = "Depth Threshold";
    ui_tooltip          = "Adjust the threshold for depth differences to count as an edge.";
> = 0.1f;

uniform float _NormalThreshold <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 5.0f;
    ui_type             = "slider";
    ui_label            = "Normal Threshold";
    ui_tooltip          = "Adjust the threshold for normal differences to count as an edge.";
> = 0.1f;

uniform float _AlphaDropOff <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "slider";
    ui_label            = "AlphaDropOff";
    ui_tooltip          = "Adjust the threshold for normal differences to count as an edge.";
> = 0.1f;

uniform float3 _Color <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "color";
    ui_label            = "Color";
    ui_tooltip          = "Adjust the threshold for normal differences to count as an edge.";
> = 0.1f;

uniform bool _TaFilterSpectrumPreview <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_label            = "TA Spectrum";
    ui_tooltip          = "Preview color spectrum changes";
> = false;


uniform bool _TaFilterTrianopy <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_label            = "Trianopy aid filter";
    ui_tooltip          = "Apply Trianopy aid filter";
> = false;

uniform bool _TaFilterSimulateTrianopy <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_label            = "Simulate Trianopy";
    ui_tooltip          = "Apply Trianopy simulation filter";
> = false;

texture2D NormalTex {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D Normals {
    Texture     = NormalTex;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

texture2D EgesTex {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D Edges {
    Texture     = EgesTex;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

texture2D DeTriTex {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D DeTri {
    Texture     = DeTriTex;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

//
// As defined per reshade documentation
#ifndef RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
	#define RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 10000.0
#endif

texture DepthBufferTex : DEPTH;
texture ColorBufferTex : COLOR;

sampler DepthBuffer {
    Texture = DepthBufferTex;
};

sampler ColorBuffer {
    Texture = ColorBufferTex;
};

//
// Simplified as per reshade documentation
float GetLinearizedDepth(float2 texcoord) {
    float depth = tex2Dlod(DepthBuffer, float4(texcoord, 0, 0)).x;
    depth /= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - depth * (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - 1);

    return depth;
}

//
// Creates normal texture. It's own pass
float4 PS_CalculateNormals(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    float3 texelSize = float3(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT, 0.0);
	float2 posCenter = uv;
	float2 posNorth  = posCenter - texelSize.zy;
	float2 posEast   = posCenter + texelSize.xz; 

    float centerDepth = GetLinearizedDepth(posCenter);

	float3 vertCenter = float3(posCenter - 0.5, 1) * centerDepth;
	float3 vertNorth  = float3(posNorth - 0.5,  1) * GetLinearizedDepth(posNorth);
	float3 vertEast   = float3(posEast - 0.5,   1) * GetLinearizedDepth(posEast);

	return float4(normalize(cross(vertCenter - vertNorth, vertCenter - vertEast)), centerDepth);

}

//
// Runs on edge detection pass
float4 PS_EdgeDetect(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    float2 texelSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

    float4 c  = tex2D(Normals, uv + float2( 0,  0) * texelSize);
    float4 w  = tex2D(Normals, uv + float2(-1,  0) * texelSize);
    float4 e  = tex2D(Normals, uv + float2( 1,  0) * texelSize);
    float4 n  = tex2D(Normals, uv + float2( 0, -1) * texelSize);
    float4 s  = tex2D(Normals, uv + float2( 0,  1) * texelSize);
    float4 nw = tex2D(Normals, uv + float2(-1, -1) * texelSize);
    float4 sw = tex2D(Normals, uv + float2( 1, -1) * texelSize);
    float4 ne = tex2D(Normals, uv + float2(-1,  1) * texelSize);
    float4 se = tex2D(Normals, uv + float2( 1,  1) * texelSize);

    float depthSum = 0.0f;
    //depthSum += abs(w.w  - c.w);
    //depthSum += abs(e.w  - c.w);
    //depthSum += abs(n.w  - c.w);
    //depthSum += abs(s.w  - c.w);
    //depthSum += abs(nw.w - c.w);
    //depthSum += abs(sw.w - c.w);
    //depthSum += abs(ne.w - c.w);
    //depthSum += abs(se.w - c.w);
    

    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2(-1,  0) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2( 1,  0) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2( 0, -1) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2( 0,  1) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2(-1, -1) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2( 1, -1) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2(-1,  1) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);
    depthSum += abs(tex2Dlod(ReShade::DepthBuffer, float4(uv + float2( 1,  1) * texelSize, 0, 0)).x - tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x);

    float depth = tex2Dlod(ReShade::DepthBuffer, float4(uv, 0, 0)).x;
    depth /= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - depth * (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - 1);
    depth = 1 - depth / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    //depth /= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    
    //depth = (exp(depth * log(0.01 + 1.0)) - 1.0) / 0.01;
    //depth = 1 / depth;
    //depthSum = 1 / depth;

    //depthSum *= 10000;
    
    //if (true)
    //    return float4(depthSum, depthSum, depthSum, 1.0f);


    float3 normalSum = 0.0f;
    normalSum += abs(w.rgb  - c.rgb);
    normalSum += abs(e.rgb  - c.rgb);
    normalSum += abs(n.rgb  - c.rgb);
    normalSum += abs(s.rgb  - c.rgb);
    normalSum += abs(nw.rgb - c.rgb);
    normalSum += abs(sw.rgb - c.rgb);
    normalSum += abs(ne.rgb - c.rgb);
    normalSum += abs(se.rgb - c.rgb);

    float alpha = 0;
    float4 output = float4(0, 0, 0, 0);
    if (dot(normalSum, 1) > _NormalThreshold && depthSum > _DepthThreshold / 10000)
    {
        output.r = _Color.r;
        output.g = _Color.g;
        output.b = _Color.b;
        output.a = 1;// - depth * _AlphaDropOff;// - depth * _AlphaDropOff;
    }
    
    return output;
}

float3 rgb_to_hsv(float3 rgb) {
    float r = rgb.r;
    float g = rgb.g;
    float b = rgb.b;

    float M = max(max(r,g),b);
    float m = min(min(r,g),b);

    float c = M - m;

    float h, s, v;

    s = c / M;

    float R, G, B;

    R = (M-r) / c;
    G = (M-g) / c;
    B = (M-b) / c;

    if (M == m) h = 0;
    else if (M == r) h = 0.0f + B - G;
    else if (M == g) h = 2 + R - B;
    else h = 4 + G - R;

    h = frac(h / 6.0f) * 360;

    v = M;

    return float3(h,s,v);
}

float3 hsv_to_rgb(float3 hsv) {
    float h = hsv.r;
    float s = hsv.g;
    float v = hsv.b;

    float C = v * s;

    float X = C * (1 - abs( frac(h / 60 / 2) * 2 - 1));
    float m = v - C;

    float R = 0.0f, G = 0.0f, B = 0.0f;

    if (h < 60) {
        R = C;
        G = X;
    } else if (h < 120) {
        R = X;
        G = C;
    } else if (h < 180) {
        G = C;
        B = X;
    } else if (h < 240) {
        G = X;
        B = C;
    } else if (h < 300) {
        R = X;
        B = C;
    } else {
        R = C;
        B = X;
    }

    R = R + m;
    G = G + m;
    B = B + m;

    return float3(R, G, B);

}

float remap_r_t_r(float h,float r1min,float r1max,float r2min,float r2max)
{
    return (h-r1min) / (r1max - r1min) * (r2max -r2min) + r2min;
}

float h_filter(float h) {
    if (h < 40) return remap_r_t_r(h, 0, 40, 0, 20);
    if (h < 50) return remap_r_t_r(h, 40, 50, 20, 100);
    if (h < 80) return remap_r_t_r(h, 50, 80, 100, 150);
    if (h < 140) return remap_r_t_r(h, 80, 140, 150, 200);
    if (h < 280) return remap_r_t_r(h, 140, 280, 200, 230);
    if (h < 340) return remap_r_t_r(h, 280, 340, 230, 280);
    return remap_r_t_r(h, 340, 360, 280, 360);
}

// https://github.com/DaltonLens/libDaltonLens/blob/master/libDaltonLens.c
float3 rgb_trian(float3 rgb) {
    float3 n = float3(0.03901f, -0.02788f, -0.01113f);
    float dot_w_sep = dot(rgb,n);
    float3x3 rgbCvdFromRgb;
    if (dot_w_sep >= 0) {
        rgbCvdFromRgb = float3x3(
            1.01277, 0.13548, -0.14826,
            -0.01243, 0.86812, 0.14431,
            0.07589, 0.80500, 0.11911
        );
    } else {
        rgbCvdFromRgb = float3x3(
            0.93678, 0.18979, -0.12657,
            0.06154, 0.81526, 0.12320,
            -0.37562, 1.12767, 0.24796
        );
    }

    float3 rgb_cvd = mul(rgbCvdFromRgb,rgb);

    return rgb_cvd;
}

float4 PS_Detri(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {

    // float4 0;

    float3 o;

    if (_TaFilterSpectrumPreview) {
        if (uv.y < 0.25f) //regular
            o = hsv_to_rgb(float3(uv.x * 360.0f,1,1));
        else if (uv.y < 0.5f) // regular trian
            o = rgb_trian(hsv_to_rgb(float3(uv.x * 360.0f,1,1)));
        else if (uv.y < 0.75f)
            o = hsv_to_rgb(float3(h_filter(uv.x * 360.0f),1,1));
        else 
            o = rgb_trian(hsv_to_rgb(float3(h_filter(uv.x * 360.0f),1,1)));
    } else {
        float4 p_rgba = tex2D(ColorBuffer, uv);

        o = p_rgba.rgb;

        if (_TaFilterTrianopy) {
            float3 p_hsv = rgb_to_hsv(o);
            o = hsv_to_rgb(float3(h_filter(p_hsv.x), p_hsv.y, p_hsv.z));
        }
        if (_TaFilterSimulateTrianopy) {
            o = rgb_trian(o);
        }
    }

    
    return float4(o,1);
}

technique E_DET < ui_label = "_E_DET"; ui_tooltip = "Replaces the screen image with an edges image."; > {
    pass {
        RenderTarget = NormalTex;
        VertexShader = EdgePostProcessVS;
        PixelShader = PS_CalculateNormals;
    }

    pass {
        RenderTarget = EgesTex;
        VertexShader = EdgePostProcessVS;
        PixelShader = PS_EdgeDetect;
    }

    pass {
        RenderTarget = DeTriTex;
        VertexShader = PostProcessVS;
        PixelShader = PS_Detri;
    }
}

