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
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 0.1f;
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
    ui_tooltip          = "";
> = 0.1f;

uniform float _AlphaMultiplier <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "slider";
    ui_label            = "Alpha Multiplier";
    ui_tooltip          = "";
> = 0.1f;

uniform bool _DrawEdges <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_type             = "box";
    ui_label            = "Draw Edges";
    ui_tooltip          = "";
> = 0.1f;

uniform bool _DLAAEdges <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_type             = "box";
    ui_label            = "DLAA Edges";
    ui_tooltip          = "";
> = 0.1f;

uniform float _AlphaDLAAThreshold <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "slider";
    ui_label            = "DLAA Alpha Threshold";
    ui_tooltip          = "";
> = 0.1f;

uniform float3 _Color <
    ui_category         = "Preprocess Settings";
    ui_category_closed  = true;
    ui_min              = 0.0f;
    ui_max              = 1.0f;
    ui_type             = "color";
    ui_label            = "Color";
    ui_tooltip          = "";
> = 0.1f;

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

//
// As defined per reshade documentation
#ifndef RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
	#define RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 10000.0
#endif

texture DepthBufferTex : DEPTH;

sampler DepthBuffer {
    Texture = DepthBufferTex;
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

    float4 color = tex2D(BackBuffer, uv + float2( 0,  0) * texelSize);
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

technique E_DET < ui_label = "_E_DET"; ui_tooltip = "Replaces the screen image with an edges image."; > {
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

