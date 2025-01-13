 ////--------//
 ///**DLAA**///
 //--------////

 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 //* Directionally localized antialiasing.                                     																									
 //* For Reshade 3.0																																								
 //* --------------------------																																					
 //* This work is licensed under a Creative Commons Attribution 3.0 Unported License.																							
 //* So you are free to share, modify and adapt it for your needs, and even use it for commercial use.																		
 //* I would also love to hear about a project you are using it with.																											
 //* https://creativecommons.org/licenses/by/3.0/us/																																
 //*																																												
 //* Have fun,																																								
 //* Jose Negrete AKA BlueSkyDefender																																			
 //*																																										
 //* http://and.intercon.ru/releases/talks/dlaagdc2011/																														
 //* ---------------------------------																																			
 //*                              
 //* Directionally Localized Anti-Aliasing (DLAA)
 //* Original method by Dmitry Andreev - Copyright (C) LucasArts 2010-2011                                              																								
 //* 																																											
 //* Lightly optimized by Marot Satil for the GShade project.
 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////
// Modified by Haansu_
////

uniform int View_Mode <
	ui_type = "combo";
	ui_items = "DLAA Out\0Mask View A\0Mask View B\0";
	ui_label = "View Mode";
	ui_tooltip = "This is used to select the normal view output or debug view.";
> = 0;

/////////////////////////////////////////////////////D3D Starts Here/////////////////////////////////////////////////////////////////
#define pix float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define lambda 3.0f
#define epsilon 0.1f

texture SLPtex {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
																				
sampler SamplerLoadedPixel {
	Texture = SLPtex;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Luminosity Intensity
float LI(in float3 value)
{	
	//Luminosity Controll from 0.1 to 1.0 
	//If GGG value of 0.333, 0.333, 0.333 is about right for Green channel. 
	//Slide 51 talk more about this.	
	return dot(value.ggg,float3(0.333, 0.333, 0.333));
}

float4 LP(sampler2D prcTex, float2 tc, float dx, float dy) {
	return tex2D(prcTex, tc + float2(dx, dy) * pix.xy);
}

float4 PreFilter(sampler input, float4 position, float2 texcoord)
{

    const float4 center = LP(input, texcoord,  0,  0);
    const float4 left   = LP(input, texcoord, -1,  0);
    const float4 right  = LP(input, texcoord,  1,  0);
    const float4 top    = LP(input, texcoord,  0, -1);
    const float4 bottom = LP(input, texcoord,  0,  1);

    const float4 edges = 4.0 * abs((left + right + top + bottom) - 4.0 * center);
    const float  edgesLum = LI(edges.rgb);

    return float4(center.rgb, edgesLum);
}

float4 SLP(sampler2D prcTex, float2 tc, float dx, float dy) {
	return tex2D(prcTex, tc + float2(dx, dy) * pix.xy);
}

//Information on Slide 44 says to run the edge processing jointly short and Large.
float4 PrcDLAA(sampler2D input, sampler2D output, float4 position, float2 texcoord) {
	float4 DLAA, DLAA_S, DLAA_L;
	
	//5 bi-linear samples cross
	const float4 Center = SLP(output, texcoord,    0,    0);
	const float4 Left   = SLP(output, texcoord, -1.0,    0);
	const float4 Right  = SLP(output, texcoord,  1.0,    0);
	const float4 Up     = SLP(output, texcoord,    0, -1.0);
	const float4 Down   = SLP(output, texcoord,    0,  1.0);  

	//Combine horizontal and vertical blurs together
	const float4 combH = 2.0 * ( Left + Right );
	const float4 combV = 2.0 * ( Up + Down );
	
	//Bi-directional anti-aliasing using HORIZONTAL & VERTICAL blur and horizontal edge detection
	//Slide information triped me up here. Read slide 43.
	//Edge detection
	const float4 CenterDiffH	= abs( combH - 4.0 * Center ) / 4.0;
	const float4 CenterDiffV	= abs( combV - 4.0 * Center ) / 4.0;

	//Blur
	const float4 blurredH		= (combH + 2.0 * Center) / 6.0;
	const float4 blurredV		= (combV + 2.0 * Center) / 6.0;
	
	//Edge detection
	const float LumH			= LI( CenterDiffH.rgb );
	const float LumV			= LI( CenterDiffV.rgb );
	
	const float LumHB = LI(blurredH.xyz);
    const float LumVB = LI(blurredV.xyz);
    
	//t
	const float satAmountH 	= saturate( ( lambda * LumH - epsilon ) / LumVB );
	const float satAmountV 	= saturate( ( lambda * LumV - epsilon ) / LumHB );
	
	//color = lerp(color,blur,sat(Edge/blur)
	//Re-blend Short Edge Done
	DLAA = lerp( Center, blurredH, satAmountV );
	DLAA = lerp( DLAA,   blurredV, satAmountH *  0.5f);
   	
	float4  HNeg, HNegA, HNegB, HNegC, HNegD, HNegE, 
			HPos, HPosA, HPosB, HPosC, HPosD, HPosE, 
			VNeg, VNegA, VNegB, VNegC, 
			VPos, VPosA, VPosB, VPosC;
			
	// Long Edges 
    //16 bi-linear samples cross, added extra bi-linear samples in each direction.
    HNeg    = Left;
    HNegA   = SLP(output, texcoord,  -3.5 ,  0.0 );
	HNegB   = SLP(output, texcoord,  -5.5 ,  0.0 );
	HNegC   = SLP(output, texcoord,  -7.5 ,  0.0 );
	
	HPos    = Right;
	HPosA   = SLP(output, texcoord,  3.5 ,  0.0 );	
	HPosB   = SLP(output, texcoord,  5.5 ,  0.0 );
	HPosC   = SLP(output, texcoord,  7.5 ,  0.0 );
	
	VNeg    = Up;
	VNegA   = SLP(output, texcoord,  0.0,-3.5  );
	VNegB   = SLP(output, texcoord,  0.0,-5.5  );
	VNegC   = SLP(output, texcoord,  0.0,-7.5  );
	
	VPos    = Down;
	VPosA   = SLP(output, texcoord,  0.0, 3.5  );
	VPosB   = SLP(output, texcoord,  0.0, 5.5  );
	VPosC   = SLP(output, texcoord,  0.0, 7.5  );
	
    //Long Edge detection H & V
    const float4 AvgBlurH = ( HNeg + HNegA + HNegB + HNegC + HPos + HPosA + HPosB + HPosC ) / 8;   
    const float4 AvgBlurV = ( VNeg + VNegA + VNegB + VNegC + VPos + VPosA + VPosB + VPosC ) / 8;
	const float EAH = saturate( AvgBlurH.a * 2.0 - 1.0 );
	const float EAV = saturate( AvgBlurV.a * 2.0 - 1.0 );
        
	const float longEdge = abs( EAH - EAV ) + abs(LumH + LumV);
	const float Mask = longEdge > 0.2;
	//Used to Protect Text
	if ( Mask )
    {
	const float4 left		= LP(input, texcoord,-1 , 0);
	const float4 right		= LP(input, texcoord, 1 , 0);
	const float4 up			= LP(input, texcoord, 0 ,-1);
	const float4 down		= LP(input, texcoord, 0 , 1);  
            
	//Merge for BlurSamples.
    const float LongBlurLumH = LI( AvgBlurH.rgb);  
	const float LongBlurLumV = LI( AvgBlurV.rgb );
	
	const float centerLI = LI( Center.rgb );
	const float leftLI	 = LI( left.rgb );
	const float rightLI  = LI( right.rgb );
	const float upLI     = LI( up.rgb );
	const float downLI   = LI( down.rgb );
  
    const float blurUp    = saturate( 0.0 + ( LongBlurLumH - upLI    ) / (centerLI - upLI) );
    const float blurLeft  = saturate( 0.0 + ( LongBlurLumV - leftLI   ) / (centerLI - leftLI) );
    const float blurDown  = saturate( 1.0 + ( LongBlurLumH - centerLI ) / (centerLI - downLI) );
    const float blurRight = saturate( 1.0 + ( LongBlurLumV - centerLI ) / (centerLI - rightLI) );

    float4 UDLR = float4( blurLeft, blurRight, blurUp, blurDown );

    if (UDLR.r == 0.0 && UDLR.g == 0.0 && UDLR.b == 0.0 && UDLR.a == 0.0)
		UDLR = float4(1.0, 1.0, 1.0, 1.0);
	
    float4 V = lerp( left , Center, UDLR.x );
		   V = lerp( right, V	  , UDLR.y );
		       
    float4 H = lerp( up   , Center, UDLR.z );
		   H = lerp( down , H	  , UDLR.w );
	
	//Reuse short samples and DLAA Long Edge Out.
    DLAA = lerp( DLAA , V , EAV);
	DLAA = lerp( DLAA , H , EAH);  
	}
	
	if(View_Mode == 1)
	{
		DLAA = Mask * 2;
	}
	else if (View_Mode == 2)
	{
		DLAA = lerp(DLAA,float4(1,1,0,1),Mask * 2);
	}
	
	return DLAA;
}
