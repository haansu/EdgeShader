//
// As per reshade documentation
void PostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
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

//
// As defined per reshade documentation
#ifndef RESHADE_DEPTH_LINEARIZATION_FAR_PLANE
	#define RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 1000.0
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
    
    float output = 0.0f;

    float depthSum = 0.0f;
    depthSum += abs(w.w  - c.w);
    depthSum += abs(e.w  - c.w);
    depthSum += abs(n.w  - c.w);
    depthSum += abs(s.w  - c.w);
    depthSum += abs(nw.w - c.w);
    depthSum += abs(sw.w - c.w);
    depthSum += abs(ne.w - c.w);
    depthSum += abs(se.w - c.w);

    if (depthSum > _DepthThreshold)
        output = 1.0f;

    float3 normalSum = 0.0f;
    normalSum += abs(w.rgb  - c.rgb);
    normalSum += abs(e.rgb  - c.rgb);
    normalSum += abs(n.rgb  - c.rgb);
    normalSum += abs(s.rgb  - c.rgb);
    normalSum += abs(nw.rgb - c.rgb);
    normalSum += abs(sw.rgb - c.rgb);
    normalSum += abs(ne.rgb - c.rgb);
    normalSum += abs(se.rgb - c.rgb);

    if (dot(normalSum, 1) > _NormalThreshold)
        output = 1.0f;

    return float4(output, output, output, 1.0);
}

technique E_DET < ui_label = "_E_DET"; ui_tooltip = "Replaces the screen image with an edges image."; > {
    pass {
        RenderTarget = NormalTex;
        VertexShader = PostProcessVS;
        PixelShader = PS_CalculateNormals;
    }

    pass {
        RenderTarget = EgesTex;
        VertexShader = PostProcessVS;
        PixelShader = PS_EdgeDetect;
    }
}