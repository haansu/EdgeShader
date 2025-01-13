#include "Reshade.fxh"
#include "DLAA.fx"


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
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 0.1f;
    ui_type             = "slider";
    ui_label            = "Depth Threshold";
    ui_tooltip          = "Adjust the threshold for depth differences to count as an edge.";
> = 0.1f;

uniform float _NormalThreshold <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 5.0f;
    ui_type             = "slider";
    ui_label            = "Normal Threshold";
    ui_tooltip          = "Adjust the threshold for normal differences to count as an edge.";
> = 0.1f;

uniform float _AlphaDropOff <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "slider";
    ui_label            = "AlphaDropOff";
    ui_tooltip          = "";
> = 0.1f;

uniform float _AlphaMultiplier <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "slider";
    ui_label            = "Alpha Multiplier";
    ui_tooltip          = "";
> = 0.1f;

uniform bool _DrawEdges <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_type             = "box";
    ui_label            = "Draw Edges";
    ui_tooltip          = "";
> = 0.1f;

uniform bool _DLAAEdges <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_type             = "box";
    ui_label            = "DLAA Edges";
    ui_tooltip          = "";
> = 0.1f;

uniform float _AlphaDLAAThreshold <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "slider";
    ui_label            = "DLAA Alpha Threshold";
    ui_tooltip          = "";
> = 0.1f;

uniform float3 _Color <
    ui_category         = "Edge Highlight Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "color";
    ui_label            = "Color";
    ui_tooltip          = "";
> = 0.1f;


uniform bool _CDS_Enable <
    ui_category         = "Color Defficiency";
    ui_label            = "Enable Color defficiency simulation or aid filters";
    ui_category_toggle = true;
> = true;

uniform int _CDS_DeffType <
    ui_category         = "Color Defficiency";
    ui_label            = "Color perception defficiency target";
    ui_type             = "combo";
    ui_items            = "protan\0deutan\0tritan\0";
>;

uniform int _CDS_FilterType <
    ui_category         = "Color Defficiency";
    ui_label            = "Filter type";
    ui_type             = "combo";
    ui_items            = "sim\0correct\0correct and sim\0";
>;

uniform float3 _CDS_Blue_Key <
    ui_category         = "Color Defficiency";
    ui_label            = "blue fixed point (for protan and deutan)";
    ui_type             = "color";
>;

uniform float3 _CDS_Yellow_Key <
    ui_category         = "Color Defficiency";
    ui_label            = "yellow fixed point (for protan and deutan)";
    ui_type             = "color";
>;

uniform float3 _CDS_Tritan_Key <
    ui_category         = "Color Defficiency";
    ui_label            = "Tritan filter fixed point (set this to cyan or green)";
    ui_type             = "color";
>;


texture BackBufferTex : COLOR;

sampler BackBuffer { 
    Texture = BackBufferTex;
};

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

texture2D EdgesTex {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D Edges {
    Texture     = EdgesTex;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

texture2D DLAATex {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D DLAA {
    Texture     = DLAATex;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

texture2D DLAATex2 {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D DLAA2 {
    Texture     = DLAATex2;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

texture2D WorldDLAATex {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D WorldDLAA {
    Texture     = WorldDLAATex;
    MagFilter   = POINT;
    MinFilter   = POINT;
    MipFilter   = POINT;
};

texture2D WorldDLAATex2 {
    Width       = BUFFER_WIDTH;
    Height      = BUFFER_HEIGHT;
    Format      = RGBA16F;
};

sampler2D WorldDLAA2 {
    Texture     = WorldDLAATex2;
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
    Texture = BackBufferTex;
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
    if (dot(normalSum, 1) > _NormalThreshold && depthSum > _DepthThreshold / 10000) {
        output.r = _Color.r;
        output.g = _Color.g;
        output.b = _Color.b;
        output.a = 1;// - depth * _AlphaDropOff;// - depth * _AlphaDropOff;
    }
    
    return output;
}

float4 PS_EdgePrefilter(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    return PreFilter(Edges, position, uv);
}

float4 PS_EdgeDLAA(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    return PrcDLAA(Edges, DLAA, position, uv);
}

float4 PS_WorldPrefilter(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    return PreFilter(BackBuffer, position, uv);
}

float4 PS_WorldDLAA(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    return PrcDLAA(BackBuffer, WorldDLAA, position, uv);
}

float4 PS_Out(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {
    float2 texelSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

    float4 color = tex2D(DeTri, uv + float2( 0,  0) * texelSize);
    //float4 color = float4(0, 0, 0, 1);
    float4 colorEDGE = tex2D(Edges, uv + float2( 0,  0) * texelSize);
    float4 colorDLAA = tex2D(DLAA2, uv + float2( 0,  0) * texelSize);

    float4 colorWorldDLAA = tex2D(WorldDLAA2, uv + float2( 0,  0) * texelSize);


    if (_DrawEdges && colorEDGE.a > 0) {
        color = lerp(color, colorEDGE, _AlphaMultiplier);
    }
    if (_DrawEdges && colorDLAA.a > 0) {
        color = lerp(color, colorDLAA, _AlphaMultiplier * colorDLAA.a);
    }
    
    if (_DLAAEdges && colorWorldDLAA.a > _AlphaDLAAThreshold) {
        float avg = (colorWorldDLAA.r + colorWorldDLAA.g + colorWorldDLAA.b) / 3;
        float4 fcolor = float4(_Color.r, _Color.g, _Color.b, 1.0f);
        color = lerp(color, fcolor, _AlphaMultiplier);
    }

    return color;
}

// https://daltonlens.org/understanding-cvd-simulation/
float3 rgb_to_xyz(float3 rgb){
    float3x3 conv_mat = float3x3(
            40.9568 , 35.5041 , 17.9167,
            21.3389 , 70.6743 ,  7.9868 ,
            1.86297, 11.462  , 91.2367
        );
    
    return mul(conv_mat, rgb);
}

float3 xyz_to_lms(float3 xyz) {
    float3x3 conv_mat = float3x3(
            0.15514 ,   0.54312 , -0.03286,
            -0.15514 ,  0.45684 , 0.03286,
            0,          0 ,       0.01608
        );
    return mul(conv_mat, xyz);
}

float3 rgb_to_lms(float3 rgb) {
    return xyz_to_lms(rgb_to_xyz(rgb));
}

float3 lms_to_xyz(float3 lms) {
    float3x3 conv_mat = float3x3(
        2.94481 , -3.50098 , 13.1722,
        1.00004 , 1.00004 , 0,
        0 , 0 , 62.1891
        );
    return mul(conv_mat, lms);
}

float3 xyz_to_rgb(float3 xyz) {
    float3x3 conv_mat = float3x3(
        0.0328041   , -0.0156571    , -0.00507134,
        -0.00997051 , 0.019112      , 0.000284916,
        0.000582757 , -0.00208132   , 0.0110283
        );
    return mul(conv_mat, xyz);
}

float3 lms_to_rgb(float3 lms) {
    return xyz_to_rgb(lms_to_xyz(lms));
}

float3x3 get_sim_matrix_protan(float3 blue_rgb, float3 yellow_rgb) {
    float3 blue_lms = rgb_to_lms(blue_rgb);
    float3 yellow_lms = rgb_to_lms(yellow_rgb);
    float3 n = cross(yellow_lms, blue_lms);
    return float3x3(
        0, -n.y/n.x, -n.z/n.x,
        0,1,0,
        0,0,1
    );
}

float3x3 get_sim_matrix_deutan(float3 blue_rgb, float3 yellow_rgb) {
    float3 blue_lms = rgb_to_lms(blue_rgb);
    float3 yellow_lms = rgb_to_lms(yellow_rgb);
    float3 n = cross(yellow_lms, blue_lms);
    return float3x3(
        1,0,0,
        -n.x/n.y,0,-n.z/n.y,
        0,0,1
    );
}

float3x3 get_sim_matrix_tritan(float3 key_rgb) {
    float3 white_lms = rgb_to_lms(float3(1,1,1));
    float3 key_lms = rgb_to_lms(key_rgb);

    float3 n = cross(white_lms, key_lms);

    return float3x3(
        1,0,0,
        0,1,0,
        -n.x/n.z,-n.y/n.z,0
    );
}

float3 simulate(float3 real, float3x3 sim_matrix) {
    return lms_to_rgb(mul(sim_matrix, rgb_to_lms(real)));
}

float3 correct(float3 real, float3x3 sim_matrix) {
    float3 percieved = simulate(real, sim_matrix);
    float3x3 err_mod = float3x3(
        0,0,0,
        0.7,1,0,
        0.7,0,1
    );
    float3 err = real - percieved;
    return mul(err_mod, err) + real; 
}

float3 correct_and_simulate(float3 real, float3x3 sim_matrix) {
    float3 corrected = correct(real, sim_matrix);

    return simulate(corrected, sim_matrix);
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



float4 PS_CD(float4 position : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET {


    float4 p_rgba = tex2D(ColorBuffer, uv);

    if(!_CDS_Enable) return p_rgba;

    float3x3 sim_matrix;

    switch(_CDS_DeffType) {
        case 0: sim_matrix = get_sim_matrix_protan(_CDS_Blue_Key, _CDS_Yellow_Key); break;
        case 1: sim_matrix = get_sim_matrix_deutan(_CDS_Blue_Key, _CDS_Yellow_Key); break;
        case 2: sim_matrix = get_sim_matrix_tritan(_CDS_Tritan_Key); break;
    }

    switch(_CDS_FilterType) {
        case 0: return float4(simulate(p_rgba.xyz, sim_matrix), p_rgba.w);
        case 1: return float4(correct(p_rgba.xyz, sim_matrix), p_rgba.w);
        case 2: return float4(correct_and_simulate(p_rgba.xyz, sim_matrix), p_rgba.w);
        default: return float4(lms_to_rgb(rgb_to_lms(p_rgba.xyz)), 1);
    }
}

technique E_DET < ui_label = "_E_DET"; ui_tooltip = "Replaces the screen image with an edges image."; > {
    pass {
        RenderTarget = DeTriTex;
        VertexShader = PostProcessVS;
        PixelShader = PS_CD;
    }

    pass {
        VertexShader = EdgePostProcessVS;
        PixelShader = PS_CalculateNormals;
        RenderTarget = NormalTex;
    }

    pass {
        VertexShader = EdgePostProcessVS;
        PixelShader  = PS_EdgeDetect;
        RenderTarget = EdgesTex;
    }

    pass {
		VertexShader = EdgePostProcessVS;
		PixelShader  = PS_EdgePrefilter;
		RenderTarget = DLAATex;
	}
	
    pass {
		VertexShader = EdgePostProcessVS;
		PixelShader  = PS_EdgeDLAA;
        RenderTarget = DLAATex2;
	}

    pass {
		VertexShader = EdgePostProcessVS;
		PixelShader  = PS_WorldPrefilter;
		RenderTarget = WorldDLAATex;
	}
	
    pass {
		VertexShader = EdgePostProcessVS;
		PixelShader  = PS_WorldDLAA;
        RenderTarget = WorldDLAATex2;
	}

    pass {
        VertexShader = EdgePostProcessVS;
		PixelShader = PS_Out;
    }
}

