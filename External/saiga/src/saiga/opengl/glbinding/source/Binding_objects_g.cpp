
#include "Binding_pch.h"


using namespace gl;


namespace glbinding
{
Function<GLuint, GLsizei> Binding::GenAsyncMarkersSGIX("glGenAsyncMarkersSGIX");
Function<void, GLsizei, GLuint*> Binding::GenBuffers("glGenBuffers");
Function<void, GLsizei, GLuint*> Binding::GenBuffersARB("glGenBuffersARB");
Function<void, GLsizei, GLuint*> Binding::GenFencesAPPLE("glGenFencesAPPLE");
Function<void, GLsizei, GLuint*> Binding::GenFencesNV("glGenFencesNV");
Function<GLuint, GLuint> Binding::GenFragmentShadersATI("glGenFragmentShadersATI");
Function<void, GLsizei, GLuint*> Binding::GenFramebuffers("glGenFramebuffers");
Function<void, GLsizei, GLuint*> Binding::GenFramebuffersEXT("glGenFramebuffersEXT");
Function<GLuint, GLsizei> Binding::GenLists("glGenLists");
Function<void, GLenum, GLuint, GLuint*> Binding::GenNamesAMD("glGenNamesAMD");
Function<void, GLsizei, GLuint*> Binding::GenOcclusionQueriesNV("glGenOcclusionQueriesNV");
Function<GLuint, GLsizei> Binding::GenPathsNV("glGenPathsNV");
Function<void, GLsizei, GLuint*> Binding::GenPerfMonitorsAMD("glGenPerfMonitorsAMD");
Function<void, GLsizei, GLuint*> Binding::GenProgramPipelines("glGenProgramPipelines");
Function<void, GLsizei, GLuint*> Binding::GenProgramsARB("glGenProgramsARB");
Function<void, GLsizei, GLuint*> Binding::GenProgramsNV("glGenProgramsNV");
Function<void, GLsizei, GLuint*> Binding::GenQueries("glGenQueries");
Function<void, GLsizei, GLuint*> Binding::GenQueriesARB("glGenQueriesARB");
Function<void, GLsizei, GLint*> Binding::GenQueryResourceTagNV("glGenQueryResourceTagNV");
Function<void, GLsizei, GLuint*> Binding::GenRenderbuffers("glGenRenderbuffers");
Function<void, GLsizei, GLuint*> Binding::GenRenderbuffersEXT("glGenRenderbuffersEXT");
Function<void, GLsizei, GLuint*> Binding::GenSamplers("glGenSamplers");
Function<void, GLsizei, GLuint*> Binding::GenSemaphoresEXT("glGenSemaphoresEXT");
Function<GLuint, GLenum, GLenum, GLenum, GLuint> Binding::GenSymbolsEXT("glGenSymbolsEXT");
Function<void, GLsizei, GLuint*> Binding::GenTextures("glGenTextures");
Function<void, GLsizei, GLuint*> Binding::GenTexturesEXT("glGenTexturesEXT");
Function<void, GLsizei, GLuint*> Binding::GenTransformFeedbacks("glGenTransformFeedbacks");
Function<void, GLsizei, GLuint*> Binding::GenTransformFeedbacksNV("glGenTransformFeedbacksNV");
Function<void, GLsizei, GLuint*> Binding::GenVertexArrays("glGenVertexArrays");
Function<void, GLsizei, GLuint*> Binding::GenVertexArraysAPPLE("glGenVertexArraysAPPLE");
Function<GLuint, GLuint> Binding::GenVertexShadersEXT("glGenVertexShadersEXT");
Function<void, GLenum> Binding::GenerateMipmap("glGenerateMipmap");
Function<void, GLenum> Binding::GenerateMipmapEXT("glGenerateMipmapEXT");
Function<void, GLenum, GLenum> Binding::GenerateMultiTexMipmapEXT("glGenerateMultiTexMipmapEXT");
Function<void, GLuint> Binding::GenerateTextureMipmap("glGenerateTextureMipmap");
Function<void, GLuint, GLenum> Binding::GenerateTextureMipmapEXT("glGenerateTextureMipmapEXT");
Function<void, GLuint, GLuint, GLenum, GLint*> Binding::GetActiveAtomicCounterBufferiv(
    "glGetActiveAtomicCounterBufferiv");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, GLchar*> Binding::GetActiveAttrib(
    "glGetActiveAttrib");
Function<void, GLhandleARB, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, GLcharARB*> Binding::GetActiveAttribARB(
    "glGetActiveAttribARB");
Function<void, GLuint, GLenum, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetActiveSubroutineName(
    "glGetActiveSubroutineName");
Function<void, GLuint, GLenum, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetActiveSubroutineUniformName(
    "glGetActiveSubroutineUniformName");
Function<void, GLuint, GLenum, GLuint, GLenum, GLint*> Binding::GetActiveSubroutineUniformiv(
    "glGetActiveSubroutineUniformiv");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, GLchar*> Binding::GetActiveUniform(
    "glGetActiveUniform");
Function<void, GLhandleARB, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, GLcharARB*> Binding::GetActiveUniformARB(
    "glGetActiveUniformARB");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetActiveUniformBlockName(
    "glGetActiveUniformBlockName");
Function<void, GLuint, GLuint, GLenum, GLint*> Binding::GetActiveUniformBlockiv("glGetActiveUniformBlockiv");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetActiveUniformName("glGetActiveUniformName");
Function<void, GLuint, GLsizei, const GLuint*, GLenum, GLint*> Binding::GetActiveUniformsiv("glGetActiveUniformsiv");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLsizei*, GLenum*, GLchar*> Binding::GetActiveVaryingNV(
    "glGetActiveVaryingNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetArrayObjectfvATI("glGetArrayObjectfvATI");
Function<void, GLenum, GLenum, GLint*> Binding::GetArrayObjectivATI("glGetArrayObjectivATI");
Function<void, GLhandleARB, GLsizei, GLsizei*, GLhandleARB*> Binding::GetAttachedObjectsARB("glGetAttachedObjectsARB");
Function<void, GLuint, GLsizei, GLsizei*, GLuint*> Binding::GetAttachedShaders("glGetAttachedShaders");
Function<GLint, GLuint, const GLchar*> Binding::GetAttribLocation("glGetAttribLocation");
Function<GLint, GLhandleARB, const GLcharARB*> Binding::GetAttribLocationARB("glGetAttribLocationARB");
Function<void, GLenum, GLuint, GLboolean*> Binding::GetBooleanIndexedvEXT("glGetBooleanIndexedvEXT");
Function<void, GLenum, GLuint, GLboolean*> Binding::GetBooleani_v("glGetBooleani_v");
Function<void, GLenum, GLboolean*> Binding::GetBooleanv("glGetBooleanv");
Function<void, GLenum, GLenum, GLint64*> Binding::GetBufferParameteri64v("glGetBufferParameteri64v");
Function<void, GLenum, GLenum, GLint*> Binding::GetBufferParameteriv("glGetBufferParameteriv");
Function<void, GLenum, GLenum, GLint*> Binding::GetBufferParameterivARB("glGetBufferParameterivARB");
Function<void, GLenum, GLenum, GLuint64EXT*> Binding::GetBufferParameterui64vNV("glGetBufferParameterui64vNV");
Function<void, GLenum, GLenum, void**> Binding::GetBufferPointerv("glGetBufferPointerv");
Function<void, GLenum, GLenum, void**> Binding::GetBufferPointervARB("glGetBufferPointervARB");
Function<void, GLenum, GLintptr, GLsizeiptr, void*> Binding::GetBufferSubData("glGetBufferSubData");
Function<void, GLenum, GLintptrARB, GLsizeiptrARB, void*> Binding::GetBufferSubDataARB("glGetBufferSubDataARB");
Function<void, GLenum, GLdouble*> Binding::GetClipPlane("glGetClipPlane");
Function<void, GLenum, GLfloat*> Binding::GetClipPlanefOES("glGetClipPlanefOES");
Function<void, GLenum, GLfixed*> Binding::GetClipPlanexOES("glGetClipPlanexOES");
Function<void, GLenum, GLenum, GLenum, void*> Binding::GetColorTable("glGetColorTable");
Function<void, GLenum, GLenum, GLenum, void*> Binding::GetColorTableEXT("glGetColorTableEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetColorTableParameterfv("glGetColorTableParameterfv");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetColorTableParameterfvEXT("glGetColorTableParameterfvEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetColorTableParameterfvSGI("glGetColorTableParameterfvSGI");
Function<void, GLenum, GLenum, GLint*> Binding::GetColorTableParameteriv("glGetColorTableParameteriv");
Function<void, GLenum, GLenum, GLint*> Binding::GetColorTableParameterivEXT("glGetColorTableParameterivEXT");
Function<void, GLenum, GLenum, GLint*> Binding::GetColorTableParameterivSGI("glGetColorTableParameterivSGI");
Function<void, GLenum, GLenum, GLenum, void*> Binding::GetColorTableSGI("glGetColorTableSGI");
Function<void, GLenum, GLenum, GLenum, GLenum, GLfloat*> Binding::GetCombinerInputParameterfvNV(
    "glGetCombinerInputParameterfvNV");
Function<void, GLenum, GLenum, GLenum, GLenum, GLint*> Binding::GetCombinerInputParameterivNV(
    "glGetCombinerInputParameterivNV");
Function<void, GLenum, GLenum, GLenum, GLfloat*> Binding::GetCombinerOutputParameterfvNV(
    "glGetCombinerOutputParameterfvNV");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetCombinerOutputParameterivNV(
    "glGetCombinerOutputParameterivNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetCombinerStageParameterfvNV("glGetCombinerStageParameterfvNV");
Function<GLuint, GLenum, GLuint> Binding::GetCommandHeaderNV("glGetCommandHeaderNV");
Function<void, GLenum, GLenum, GLint, void*> Binding::GetCompressedMultiTexImageEXT("glGetCompressedMultiTexImageEXT");
Function<void, GLenum, GLint, void*> Binding::GetCompressedTexImage("glGetCompressedTexImage");
Function<void, GLenum, GLint, void*> Binding::GetCompressedTexImageARB("glGetCompressedTexImageARB");
Function<void, GLuint, GLint, GLsizei, void*> Binding::GetCompressedTextureImage("glGetCompressedTextureImage");
Function<void, GLuint, GLenum, GLint, void*> Binding::GetCompressedTextureImageEXT("glGetCompressedTextureImageEXT");
Function<void, GLuint, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLsizei, void*>
    Binding::GetCompressedTextureSubImage("glGetCompressedTextureSubImage");
Function<void, GLenum, GLenum, GLenum, void*> Binding::GetConvolutionFilter("glGetConvolutionFilter");
Function<void, GLenum, GLenum, GLenum, void*> Binding::GetConvolutionFilterEXT("glGetConvolutionFilterEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetConvolutionParameterfv("glGetConvolutionParameterfv");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetConvolutionParameterfvEXT("glGetConvolutionParameterfvEXT");
Function<void, GLenum, GLenum, GLint*> Binding::GetConvolutionParameteriv("glGetConvolutionParameteriv");
Function<void, GLenum, GLenum, GLint*> Binding::GetConvolutionParameterivEXT("glGetConvolutionParameterivEXT");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetConvolutionParameterxvOES("glGetConvolutionParameterxvOES");
Function<void, GLsizei, GLfloat*> Binding::GetCoverageModulationTableNV("glGetCoverageModulationTableNV");
Function<GLuint, GLuint, GLsizei, GLenum*, GLenum*, GLuint*, GLenum*, GLsizei*, GLchar*> Binding::GetDebugMessageLog(
    "glGetDebugMessageLog");
Function<GLuint, GLuint, GLsizei, GLenum*, GLuint*, GLuint*, GLsizei*, GLchar*> Binding::GetDebugMessageLogAMD(
    "glGetDebugMessageLogAMD");
Function<GLuint, GLuint, GLsizei, GLenum*, GLenum*, GLuint*, GLenum*, GLsizei*, GLchar*> Binding::GetDebugMessageLogARB(
    "glGetDebugMessageLogARB");
Function<void, GLenum, GLfloat*> Binding::GetDetailTexFuncSGIS("glGetDetailTexFuncSGIS");
Function<void, GLenum, GLuint, GLdouble*> Binding::GetDoubleIndexedvEXT("glGetDoubleIndexedvEXT");
Function<void, GLenum, GLuint, GLdouble*> Binding::GetDoublei_v("glGetDoublei_v");
Function<void, GLenum, GLuint, GLdouble*> Binding::GetDoublei_vEXT("glGetDoublei_vEXT");
Function<void, GLenum, GLdouble*> Binding::GetDoublev("glGetDoublev");
Function<GLenum> Binding::GetError("glGetError");
Function<void, GLuint, GLenum, GLint*> Binding::GetFenceivNV("glGetFenceivNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetFinalCombinerInputParameterfvNV(
    "glGetFinalCombinerInputParameterfvNV");
Function<void, GLenum, GLenum, GLint*> Binding::GetFinalCombinerInputParameterivNV(
    "glGetFinalCombinerInputParameterivNV");
Function<void, GLuint*> Binding::GetFirstPerfQueryIdINTEL("glGetFirstPerfQueryIdINTEL");
Function<void, GLenum, GLfixed*> Binding::GetFixedvOES("glGetFixedvOES");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetFloatIndexedvEXT("glGetFloatIndexedvEXT");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetFloati_v("glGetFloati_v");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetFloati_vEXT("glGetFloati_vEXT");
Function<void, GLenum, GLfloat*> Binding::GetFloatv("glGetFloatv");
Function<void, GLfloat*> Binding::GetFogFuncSGIS("glGetFogFuncSGIS");
Function<GLint, GLuint, const GLchar*> Binding::GetFragDataIndex("glGetFragDataIndex");
Function<GLint, GLuint, const GLchar*> Binding::GetFragDataLocation("glGetFragDataLocation");
Function<GLint, GLuint, const GLchar*> Binding::GetFragDataLocationEXT("glGetFragDataLocationEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetFragmentLightfvSGIX("glGetFragmentLightfvSGIX");
Function<void, GLenum, GLenum, GLint*> Binding::GetFragmentLightivSGIX("glGetFragmentLightivSGIX");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetFragmentMaterialfvSGIX("glGetFragmentMaterialfvSGIX");
Function<void, GLenum, GLenum, GLint*> Binding::GetFragmentMaterialivSGIX("glGetFragmentMaterialivSGIX");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetFramebufferAttachmentParameteriv(
    "glGetFramebufferAttachmentParameteriv");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetFramebufferAttachmentParameterivEXT(
    "glGetFramebufferAttachmentParameterivEXT");
Function<void, GLenum, GLenum, GLuint, GLuint, GLsizei, GLfloat*> Binding::GetFramebufferParameterfvAMD(
    "glGetFramebufferParameterfvAMD");
Function<void, GLenum, GLenum, GLint*> Binding::GetFramebufferParameteriv("glGetFramebufferParameteriv");
Function<void, GLuint, GLenum, GLint*> Binding::GetFramebufferParameterivEXT("glGetFramebufferParameterivEXT");
Function<GLenum> Binding::GetGraphicsResetStatus("glGetGraphicsResetStatus");
Function<GLenum> Binding::GetGraphicsResetStatusARB("glGetGraphicsResetStatusARB");
Function<GLhandleARB, GLenum> Binding::GetHandleARB("glGetHandleARB");
Function<void, GLenum, GLboolean, GLenum, GLenum, void*> Binding::GetHistogram("glGetHistogram");
Function<void, GLenum, GLboolean, GLenum, GLenum, void*> Binding::GetHistogramEXT("glGetHistogramEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetHistogramParameterfv("glGetHistogramParameterfv");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetHistogramParameterfvEXT("glGetHistogramParameterfvEXT");
Function<void, GLenum, GLenum, GLint*> Binding::GetHistogramParameteriv("glGetHistogramParameteriv");
Function<void, GLenum, GLenum, GLint*> Binding::GetHistogramParameterivEXT("glGetHistogramParameterivEXT");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetHistogramParameterxvOES("glGetHistogramParameterxvOES");
Function<GLuint64, GLuint, GLint, GLboolean, GLint, GLenum> Binding::GetImageHandleARB("glGetImageHandleARB");
Function<GLuint64, GLuint, GLint, GLboolean, GLint, GLenum> Binding::GetImageHandleNV("glGetImageHandleNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetImageTransformParameterfvHP("glGetImageTransformParameterfvHP");
Function<void, GLenum, GLenum, GLint*> Binding::GetImageTransformParameterivHP("glGetImageTransformParameterivHP");
Function<void, GLhandleARB, GLsizei, GLsizei*, GLcharARB*> Binding::GetInfoLogARB("glGetInfoLogARB");
Function<GLint> Binding::GetInstrumentsSGIX("glGetInstrumentsSGIX");
Function<void, GLenum, GLuint, GLint64*> Binding::GetInteger64i_v("glGetInteger64i_v");
Function<void, GLenum, GLint64*> Binding::GetInteger64v("glGetInteger64v");
Function<void, GLenum, GLuint, GLint*> Binding::GetIntegerIndexedvEXT("glGetIntegerIndexedvEXT");
Function<void, GLenum, GLuint, GLint*> Binding::GetIntegeri_v("glGetIntegeri_v");
Function<void, GLenum, GLuint, GLuint64EXT*> Binding::GetIntegerui64i_vNV("glGetIntegerui64i_vNV");
Function<void, GLenum, GLuint64EXT*> Binding::GetIntegerui64vNV("glGetIntegerui64vNV");
Function<void, GLenum, GLint*> Binding::GetIntegerv("glGetIntegerv");
Function<void, GLenum, GLenum, GLsizei, GLenum, GLsizei, GLint*> Binding::GetInternalformatSampleivNV(
    "glGetInternalformatSampleivNV");
Function<void, GLenum, GLenum, GLenum, GLsizei, GLint64*> Binding::GetInternalformati64v("glGetInternalformati64v");
Function<void, GLenum, GLenum, GLenum, GLsizei, GLint*> Binding::GetInternalformativ("glGetInternalformativ");
Function<void, GLuint, GLenum, GLboolean*> Binding::GetInvariantBooleanvEXT("glGetInvariantBooleanvEXT");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetInvariantFloatvEXT("glGetInvariantFloatvEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetInvariantIntegervEXT("glGetInvariantIntegervEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetLightfv("glGetLightfv");
Function<void, GLenum, GLenum, GLint*> Binding::GetLightiv("glGetLightiv");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetLightxOES("glGetLightxOES");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetListParameterfvSGIX("glGetListParameterfvSGIX");
Function<void, GLuint, GLenum, GLint*> Binding::GetListParameterivSGIX("glGetListParameterivSGIX");
Function<void, GLuint, GLenum, GLboolean*> Binding::GetLocalConstantBooleanvEXT("glGetLocalConstantBooleanvEXT");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetLocalConstantFloatvEXT("glGetLocalConstantFloatvEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetLocalConstantIntegervEXT("glGetLocalConstantIntegervEXT");
Function<void, GLenum, GLuint, GLenum, GLfloat*> Binding::GetMapAttribParameterfvNV("glGetMapAttribParameterfvNV");
Function<void, GLenum, GLuint, GLenum, GLint*> Binding::GetMapAttribParameterivNV("glGetMapAttribParameterivNV");
Function<void, GLenum, GLuint, GLenum, GLsizei, GLsizei, GLboolean, void*> Binding::GetMapControlPointsNV(
    "glGetMapControlPointsNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetMapParameterfvNV("glGetMapParameterfvNV");
Function<void, GLenum, GLenum, GLint*> Binding::GetMapParameterivNV("glGetMapParameterivNV");
Function<void, GLenum, GLenum, GLdouble*> Binding::GetMapdv("glGetMapdv");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetMapfv("glGetMapfv");
Function<void, GLenum, GLenum, GLint*> Binding::GetMapiv("glGetMapiv");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetMapxvOES("glGetMapxvOES");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetMaterialfv("glGetMaterialfv");
Function<void, GLenum, GLenum, GLint*> Binding::GetMaterialiv("glGetMaterialiv");
Function<void, GLenum, GLenum, GLfixed> Binding::GetMaterialxOES("glGetMaterialxOES");
Function<void, GLuint, GLenum, GLint*> Binding::GetMemoryObjectParameterivEXT("glGetMemoryObjectParameterivEXT");
Function<void, GLenum, GLboolean, GLenum, GLenum, void*> Binding::GetMinmax("glGetMinmax");
Function<void, GLenum, GLboolean, GLenum, GLenum, void*> Binding::GetMinmaxEXT("glGetMinmaxEXT");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetMinmaxParameterfv("glGetMinmaxParameterfv");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetMinmaxParameterfvEXT("glGetMinmaxParameterfvEXT");
Function<void, GLenum, GLenum, GLint*> Binding::GetMinmaxParameteriv("glGetMinmaxParameteriv");
Function<void, GLenum, GLenum, GLint*> Binding::GetMinmaxParameterivEXT("glGetMinmaxParameterivEXT");
Function<void, GLenum, GLenum, GLenum, GLfloat*> Binding::GetMultiTexEnvfvEXT("glGetMultiTexEnvfvEXT");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetMultiTexEnvivEXT("glGetMultiTexEnvivEXT");
Function<void, GLenum, GLenum, GLenum, GLdouble*> Binding::GetMultiTexGendvEXT("glGetMultiTexGendvEXT");
Function<void, GLenum, GLenum, GLenum, GLfloat*> Binding::GetMultiTexGenfvEXT("glGetMultiTexGenfvEXT");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetMultiTexGenivEXT("glGetMultiTexGenivEXT");
Function<void, GLenum, GLenum, GLint, GLenum, GLenum, void*> Binding::GetMultiTexImageEXT("glGetMultiTexImageEXT");
Function<void, GLenum, GLenum, GLint, GLenum, GLfloat*> Binding::GetMultiTexLevelParameterfvEXT(
    "glGetMultiTexLevelParameterfvEXT");
Function<void, GLenum, GLenum, GLint, GLenum, GLint*> Binding::GetMultiTexLevelParameterivEXT(
    "glGetMultiTexLevelParameterivEXT");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetMultiTexParameterIivEXT("glGetMultiTexParameterIivEXT");
Function<void, GLenum, GLenum, GLenum, GLuint*> Binding::GetMultiTexParameterIuivEXT("glGetMultiTexParameterIuivEXT");
Function<void, GLenum, GLenum, GLenum, GLfloat*> Binding::GetMultiTexParameterfvEXT("glGetMultiTexParameterfvEXT");
Function<void, GLenum, GLenum, GLenum, GLint*> Binding::GetMultiTexParameterivEXT("glGetMultiTexParameterivEXT");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetMultisamplefv("glGetMultisamplefv");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetMultisamplefvNV("glGetMultisamplefvNV");
Function<void, GLuint, GLenum, GLint64*> Binding::GetNamedBufferParameteri64v("glGetNamedBufferParameteri64v");
Function<void, GLuint, GLenum, GLint*> Binding::GetNamedBufferParameteriv("glGetNamedBufferParameteriv");
Function<void, GLuint, GLenum, GLint*> Binding::GetNamedBufferParameterivEXT("glGetNamedBufferParameterivEXT");
Function<void, GLuint, GLenum, GLuint64EXT*> Binding::GetNamedBufferParameterui64vNV(
    "glGetNamedBufferParameterui64vNV");
Function<void, GLuint, GLenum, void**> Binding::GetNamedBufferPointerv("glGetNamedBufferPointerv");
Function<void, GLuint, GLenum, void**> Binding::GetNamedBufferPointervEXT("glGetNamedBufferPointervEXT");
Function<void, GLuint, GLintptr, GLsizeiptr, void*> Binding::GetNamedBufferSubData("glGetNamedBufferSubData");
Function<void, GLuint, GLintptr, GLsizeiptr, void*> Binding::GetNamedBufferSubDataEXT("glGetNamedBufferSubDataEXT");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetNamedFramebufferAttachmentParameteriv(
    "glGetNamedFramebufferAttachmentParameteriv");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetNamedFramebufferAttachmentParameterivEXT(
    "glGetNamedFramebufferAttachmentParameterivEXT");
Function<void, GLuint, GLenum, GLuint, GLuint, GLsizei, GLfloat*> Binding::GetNamedFramebufferParameterfvAMD(
    "glGetNamedFramebufferParameterfvAMD");
Function<void, GLuint, GLenum, GLint*> Binding::GetNamedFramebufferParameteriv("glGetNamedFramebufferParameteriv");
Function<void, GLuint, GLenum, GLint*> Binding::GetNamedFramebufferParameterivEXT(
    "glGetNamedFramebufferParameterivEXT");
Function<void, GLuint, GLenum, GLuint, GLint*> Binding::GetNamedProgramLocalParameterIivEXT(
    "glGetNamedProgramLocalParameterIivEXT");
Function<void, GLuint, GLenum, GLuint, GLuint*> Binding::GetNamedProgramLocalParameterIuivEXT(
    "glGetNamedProgramLocalParameterIuivEXT");
Function<void, GLuint, GLenum, GLuint, GLdouble*> Binding::GetNamedProgramLocalParameterdvEXT(
    "glGetNamedProgramLocalParameterdvEXT");
Function<void, GLuint, GLenum, GLuint, GLfloat*> Binding::GetNamedProgramLocalParameterfvEXT(
    "glGetNamedProgramLocalParameterfvEXT");
Function<void, GLuint, GLenum, GLenum, void*> Binding::GetNamedProgramStringEXT("glGetNamedProgramStringEXT");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetNamedProgramivEXT("glGetNamedProgramivEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetNamedRenderbufferParameteriv("glGetNamedRenderbufferParameteriv");
Function<void, GLuint, GLenum, GLint*> Binding::GetNamedRenderbufferParameterivEXT(
    "glGetNamedRenderbufferParameterivEXT");
Function<void, GLint, const GLchar*, GLsizei, GLint*, GLchar*> Binding::GetNamedStringARB("glGetNamedStringARB");
Function<void, GLint, const GLchar*, GLenum, GLint*> Binding::GetNamedStringivARB("glGetNamedStringivARB");
Function<void, GLuint, GLuint*> Binding::GetNextPerfQueryIdINTEL("glGetNextPerfQueryIdINTEL");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetObjectBufferfvATI("glGetObjectBufferfvATI");
Function<void, GLuint, GLenum, GLint*> Binding::GetObjectBufferivATI("glGetObjectBufferivATI");
Function<void, GLenum, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetObjectLabel("glGetObjectLabel");
Function<void, GLenum, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetObjectLabelEXT("glGetObjectLabelEXT");
Function<void, GLhandleARB, GLenum, GLfloat*> Binding::GetObjectParameterfvARB("glGetObjectParameterfvARB");
Function<void, GLenum, GLuint, GLenum, GLint*> Binding::GetObjectParameterivAPPLE("glGetObjectParameterivAPPLE");
Function<void, GLhandleARB, GLenum, GLint*> Binding::GetObjectParameterivARB("glGetObjectParameterivARB");
Function<void, const void*, GLsizei, GLsizei*, GLchar*> Binding::GetObjectPtrLabel("glGetObjectPtrLabel");
Function<void, GLuint, GLenum, GLint*> Binding::GetOcclusionQueryivNV("glGetOcclusionQueryivNV");
Function<void, GLuint, GLenum, GLuint*> Binding::GetOcclusionQueryuivNV("glGetOcclusionQueryuivNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetPathColorGenfvNV("glGetPathColorGenfvNV");
Function<void, GLenum, GLenum, GLint*> Binding::GetPathColorGenivNV("glGetPathColorGenivNV");
Function<void, GLuint, GLubyte*> Binding::GetPathCommandsNV("glGetPathCommandsNV");
Function<void, GLuint, GLfloat*> Binding::GetPathCoordsNV("glGetPathCoordsNV");
Function<void, GLuint, GLfloat*> Binding::GetPathDashArrayNV("glGetPathDashArrayNV");
Function<GLfloat, GLuint, GLsizei, GLsizei> Binding::GetPathLengthNV("glGetPathLengthNV");
Function<void, PathRenderingMaskNV, GLuint, GLsizei, GLsizei, GLfloat*> Binding::GetPathMetricRangeNV(
    "glGetPathMetricRangeNV");
Function<void, PathRenderingMaskNV, GLsizei, GLenum, const void*, GLuint, GLsizei, GLfloat*> Binding::GetPathMetricsNV(
    "glGetPathMetricsNV");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetPathParameterfvNV("glGetPathParameterfvNV");
Function<void, GLuint, GLenum, GLint*> Binding::GetPathParameterivNV("glGetPathParameterivNV");
Function<void, GLenum, GLsizei, GLenum, const void*, GLuint, GLfloat, GLfloat, GLenum, GLfloat*>
    Binding::GetPathSpacingNV("glGetPathSpacingNV");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetPathTexGenfvNV("glGetPathTexGenfvNV");
Function<void, GLenum, GLenum, GLint*> Binding::GetPathTexGenivNV("glGetPathTexGenivNV");
Function<void, GLuint, GLuint, GLuint, GLchar*, GLuint, GLchar*, GLuint*, GLuint*, GLuint*, GLuint*, GLuint64*>
    Binding::GetPerfCounterInfoINTEL("glGetPerfCounterInfoINTEL");
Function<void, GLuint, GLenum, GLsizei, GLuint*, GLint*> Binding::GetPerfMonitorCounterDataAMD(
    "glGetPerfMonitorCounterDataAMD");
Function<void, GLuint, GLuint, GLenum, void*> Binding::GetPerfMonitorCounterInfoAMD("glGetPerfMonitorCounterInfoAMD");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetPerfMonitorCounterStringAMD(
    "glGetPerfMonitorCounterStringAMD");
Function<void, GLuint, GLint*, GLint*, GLsizei, GLuint*> Binding::GetPerfMonitorCountersAMD(
    "glGetPerfMonitorCountersAMD");
Function<void, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetPerfMonitorGroupStringAMD(
    "glGetPerfMonitorGroupStringAMD");
Function<void, GLint*, GLsizei, GLuint*> Binding::GetPerfMonitorGroupsAMD("glGetPerfMonitorGroupsAMD");
Function<void, GLuint, GLuint, GLsizei, void*, GLuint*> Binding::GetPerfQueryDataINTEL("glGetPerfQueryDataINTEL");
Function<void, GLchar*, GLuint*> Binding::GetPerfQueryIdByNameINTEL("glGetPerfQueryIdByNameINTEL");
Function<void, GLuint, GLuint, GLchar*, GLuint*, GLuint*, GLuint*, GLuint*> Binding::GetPerfQueryInfoINTEL(
    "glGetPerfQueryInfoINTEL");
Function<void, GLenum, GLfloat*> Binding::GetPixelMapfv("glGetPixelMapfv");
Function<void, GLenum, GLuint*> Binding::GetPixelMapuiv("glGetPixelMapuiv");
Function<void, GLenum, GLushort*> Binding::GetPixelMapusv("glGetPixelMapusv");
Function<void, GLenum, GLint, GLfixed*> Binding::GetPixelMapxv("glGetPixelMapxv");
Function<void, GLenum, GLfloat*> Binding::GetPixelTexGenParameterfvSGIS("glGetPixelTexGenParameterfvSGIS");
Function<void, GLenum, GLint*> Binding::GetPixelTexGenParameterivSGIS("glGetPixelTexGenParameterivSGIS");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetPixelTransformParameterfvEXT("glGetPixelTransformParameterfvEXT");
Function<void, GLenum, GLenum, GLint*> Binding::GetPixelTransformParameterivEXT("glGetPixelTransformParameterivEXT");
Function<void, GLenum, GLuint, void**> Binding::GetPointerIndexedvEXT("glGetPointerIndexedvEXT");
Function<void, GLenum, GLuint, void**> Binding::GetPointeri_vEXT("glGetPointeri_vEXT");
Function<void, GLenum, void**> Binding::GetPointerv("glGetPointerv");
Function<void, GLenum, void**> Binding::GetPointervEXT("glGetPointervEXT");
Function<void, GLubyte*> Binding::GetPolygonStipple("glGetPolygonStipple");
Function<void, GLuint, GLsizei, GLsizei*, GLenum*, void*> Binding::GetProgramBinary("glGetProgramBinary");
Function<void, GLenum, GLuint, GLint*> Binding::GetProgramEnvParameterIivNV("glGetProgramEnvParameterIivNV");
Function<void, GLenum, GLuint, GLuint*> Binding::GetProgramEnvParameterIuivNV("glGetProgramEnvParameterIuivNV");
Function<void, GLenum, GLuint, GLdouble*> Binding::GetProgramEnvParameterdvARB("glGetProgramEnvParameterdvARB");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetProgramEnvParameterfvARB("glGetProgramEnvParameterfvARB");
Function<void, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetProgramInfoLog("glGetProgramInfoLog");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetProgramInterfaceiv("glGetProgramInterfaceiv");
Function<void, GLenum, GLuint, GLint*> Binding::GetProgramLocalParameterIivNV("glGetProgramLocalParameterIivNV");
Function<void, GLenum, GLuint, GLuint*> Binding::GetProgramLocalParameterIuivNV("glGetProgramLocalParameterIuivNV");
Function<void, GLenum, GLuint, GLdouble*> Binding::GetProgramLocalParameterdvARB("glGetProgramLocalParameterdvARB");
Function<void, GLenum, GLuint, GLfloat*> Binding::GetProgramLocalParameterfvARB("glGetProgramLocalParameterfvARB");
Function<void, GLuint, GLsizei, const GLubyte*, GLdouble*> Binding::GetProgramNamedParameterdvNV(
    "glGetProgramNamedParameterdvNV");
Function<void, GLuint, GLsizei, const GLubyte*, GLfloat*> Binding::GetProgramNamedParameterfvNV(
    "glGetProgramNamedParameterfvNV");
Function<void, GLenum, GLuint, GLenum, GLdouble*> Binding::GetProgramParameterdvNV("glGetProgramParameterdvNV");
Function<void, GLenum, GLuint, GLenum, GLfloat*> Binding::GetProgramParameterfvNV("glGetProgramParameterfvNV");
Function<void, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetProgramPipelineInfoLog("glGetProgramPipelineInfoLog");
Function<void, GLuint, GLenum, GLint*> Binding::GetProgramPipelineiv("glGetProgramPipelineiv");
Function<GLuint, GLuint, GLenum, const GLchar*> Binding::GetProgramResourceIndex("glGetProgramResourceIndex");
Function<GLint, GLuint, GLenum, const GLchar*> Binding::GetProgramResourceLocation("glGetProgramResourceLocation");
Function<GLint, GLuint, GLenum, const GLchar*> Binding::GetProgramResourceLocationIndex(
    "glGetProgramResourceLocationIndex");
Function<void, GLuint, GLenum, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetProgramResourceName(
    "glGetProgramResourceName");
Function<void, GLuint, GLenum, GLuint, GLsizei, const GLenum*, GLsizei, GLsizei*, GLfloat*>
    Binding::GetProgramResourcefvNV("glGetProgramResourcefvNV");
Function<void, GLuint, GLenum, GLuint, GLsizei, const GLenum*, GLsizei, GLsizei*, GLint*> Binding::GetProgramResourceiv(
    "glGetProgramResourceiv");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetProgramStageiv("glGetProgramStageiv");
Function<void, GLenum, GLenum, void*> Binding::GetProgramStringARB("glGetProgramStringARB");
Function<void, GLuint, GLenum, GLubyte*> Binding::GetProgramStringNV("glGetProgramStringNV");
Function<void, GLenum, GLuint, GLuint*> Binding::GetProgramSubroutineParameteruivNV(
    "glGetProgramSubroutineParameteruivNV");
Function<void, GLuint, GLenum, GLint*> Binding::GetProgramiv("glGetProgramiv");
Function<void, GLenum, GLenum, GLint*> Binding::GetProgramivARB("glGetProgramivARB");
Function<void, GLuint, GLenum, GLint*> Binding::GetProgramivNV("glGetProgramivNV");
Function<void, GLuint, GLuint, GLenum, GLintptr> Binding::GetQueryBufferObjecti64v("glGetQueryBufferObjecti64v");
Function<void, GLuint, GLuint, GLenum, GLintptr> Binding::GetQueryBufferObjectiv("glGetQueryBufferObjectiv");
Function<void, GLuint, GLuint, GLenum, GLintptr> Binding::GetQueryBufferObjectui64v("glGetQueryBufferObjectui64v");
Function<void, GLuint, GLuint, GLenum, GLintptr> Binding::GetQueryBufferObjectuiv("glGetQueryBufferObjectuiv");
Function<void, GLenum, GLuint, GLenum, GLint*> Binding::GetQueryIndexediv("glGetQueryIndexediv");
Function<void, GLuint, GLenum, GLint64*> Binding::GetQueryObjecti64v("glGetQueryObjecti64v");
Function<void, GLuint, GLenum, GLint64*> Binding::GetQueryObjecti64vEXT("glGetQueryObjecti64vEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetQueryObjectiv("glGetQueryObjectiv");
Function<void, GLuint, GLenum, GLint*> Binding::GetQueryObjectivARB("glGetQueryObjectivARB");
Function<void, GLuint, GLenum, GLuint64*> Binding::GetQueryObjectui64v("glGetQueryObjectui64v");
Function<void, GLuint, GLenum, GLuint64*> Binding::GetQueryObjectui64vEXT("glGetQueryObjectui64vEXT");
Function<void, GLuint, GLenum, GLuint*> Binding::GetQueryObjectuiv("glGetQueryObjectuiv");
Function<void, GLuint, GLenum, GLuint*> Binding::GetQueryObjectuivARB("glGetQueryObjectuivARB");
Function<void, GLenum, GLenum, GLint*> Binding::GetQueryiv("glGetQueryiv");
Function<void, GLenum, GLenum, GLint*> Binding::GetQueryivARB("glGetQueryivARB");
Function<void, GLenum, GLenum, GLint*> Binding::GetRenderbufferParameteriv("glGetRenderbufferParameteriv");
Function<void, GLenum, GLenum, GLint*> Binding::GetRenderbufferParameterivEXT("glGetRenderbufferParameterivEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetSamplerParameterIiv("glGetSamplerParameterIiv");
Function<void, GLuint, GLenum, GLuint*> Binding::GetSamplerParameterIuiv("glGetSamplerParameterIuiv");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetSamplerParameterfv("glGetSamplerParameterfv");
Function<void, GLuint, GLenum, GLint*> Binding::GetSamplerParameteriv("glGetSamplerParameteriv");
Function<void, GLuint, GLenum, GLuint64*> Binding::GetSemaphoreParameterui64vEXT("glGetSemaphoreParameterui64vEXT");
Function<void, GLenum, GLenum, GLenum, void*, void*, void*> Binding::GetSeparableFilter("glGetSeparableFilter");
Function<void, GLenum, GLenum, GLenum, void*, void*, void*> Binding::GetSeparableFilterEXT("glGetSeparableFilterEXT");
Function<void, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetShaderInfoLog("glGetShaderInfoLog");
Function<void, GLenum, GLenum, GLint*, GLint*> Binding::GetShaderPrecisionFormat("glGetShaderPrecisionFormat");
Function<void, GLuint, GLsizei, GLsizei*, GLchar*> Binding::GetShaderSource("glGetShaderSource");
Function<void, GLhandleARB, GLsizei, GLsizei*, GLcharARB*> Binding::GetShaderSourceARB("glGetShaderSourceARB");
Function<void, GLuint, GLenum, GLint*> Binding::GetShaderiv("glGetShaderiv");
Function<void, GLenum, GLfloat*> Binding::GetSharpenTexFuncSGIS("glGetSharpenTexFuncSGIS");
Function<GLushort, GLenum> Binding::GetStageIndexNV("glGetStageIndexNV");
Function<const GLubyte*, GLenum> Binding::GetString("glGetString");
Function<const GLubyte*, GLenum, GLuint> Binding::GetStringi("glGetStringi");
Function<GLuint, GLuint, GLenum, const GLchar*> Binding::GetSubroutineIndex("glGetSubroutineIndex");
Function<GLint, GLuint, GLenum, const GLchar*> Binding::GetSubroutineUniformLocation("glGetSubroutineUniformLocation");
Function<void, GLsync, GLenum, GLsizei, GLsizei*, GLint*> Binding::GetSynciv("glGetSynciv");
Function<void, GLenum, GLfloat*> Binding::GetTexBumpParameterfvATI("glGetTexBumpParameterfvATI");
Function<void, GLenum, GLint*> Binding::GetTexBumpParameterivATI("glGetTexBumpParameterivATI");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetTexEnvfv("glGetTexEnvfv");
Function<void, GLenum, GLenum, GLint*> Binding::GetTexEnviv("glGetTexEnviv");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetTexEnvxvOES("glGetTexEnvxvOES");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetTexFilterFuncSGIS("glGetTexFilterFuncSGIS");
Function<void, GLenum, GLenum, GLdouble*> Binding::GetTexGendv("glGetTexGendv");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetTexGenfv("glGetTexGenfv");
Function<void, GLenum, GLenum, GLint*> Binding::GetTexGeniv("glGetTexGeniv");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetTexGenxvOES("glGetTexGenxvOES");
Function<void, GLenum, GLint, GLenum, GLenum, void*> Binding::GetTexImage("glGetTexImage");
Function<void, GLenum, GLint, GLenum, GLfloat*> Binding::GetTexLevelParameterfv("glGetTexLevelParameterfv");
Function<void, GLenum, GLint, GLenum, GLint*> Binding::GetTexLevelParameteriv("glGetTexLevelParameteriv");
Function<void, GLenum, GLint, GLenum, GLfixed*> Binding::GetTexLevelParameterxvOES("glGetTexLevelParameterxvOES");
Function<void, GLenum, GLenum, GLint*> Binding::GetTexParameterIiv("glGetTexParameterIiv");
Function<void, GLenum, GLenum, GLint*> Binding::GetTexParameterIivEXT("glGetTexParameterIivEXT");
Function<void, GLenum, GLenum, GLuint*> Binding::GetTexParameterIuiv("glGetTexParameterIuiv");
Function<void, GLenum, GLenum, GLuint*> Binding::GetTexParameterIuivEXT("glGetTexParameterIuivEXT");
Function<void, GLenum, GLenum, void**> Binding::GetTexParameterPointervAPPLE("glGetTexParameterPointervAPPLE");
Function<void, GLenum, GLenum, GLfloat*> Binding::GetTexParameterfv("glGetTexParameterfv");
Function<void, GLenum, GLenum, GLint*> Binding::GetTexParameteriv("glGetTexParameteriv");
Function<void, GLenum, GLenum, GLfixed*> Binding::GetTexParameterxvOES("glGetTexParameterxvOES");
Function<GLuint64, GLuint> Binding::GetTextureHandleARB("glGetTextureHandleARB");
Function<GLuint64, GLuint> Binding::GetTextureHandleNV("glGetTextureHandleNV");
Function<void, GLuint, GLint, GLenum, GLenum, GLsizei, void*> Binding::GetTextureImage("glGetTextureImage");
Function<void, GLuint, GLenum, GLint, GLenum, GLenum, void*> Binding::GetTextureImageEXT("glGetTextureImageEXT");
Function<void, GLuint, GLint, GLenum, GLfloat*> Binding::GetTextureLevelParameterfv("glGetTextureLevelParameterfv");
Function<void, GLuint, GLenum, GLint, GLenum, GLfloat*> Binding::GetTextureLevelParameterfvEXT(
    "glGetTextureLevelParameterfvEXT");
Function<void, GLuint, GLint, GLenum, GLint*> Binding::GetTextureLevelParameteriv("glGetTextureLevelParameteriv");
Function<void, GLuint, GLenum, GLint, GLenum, GLint*> Binding::GetTextureLevelParameterivEXT(
    "glGetTextureLevelParameterivEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetTextureParameterIiv("glGetTextureParameterIiv");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetTextureParameterIivEXT("glGetTextureParameterIivEXT");
Function<void, GLuint, GLenum, GLuint*> Binding::GetTextureParameterIuiv("glGetTextureParameterIuiv");
Function<void, GLuint, GLenum, GLenum, GLuint*> Binding::GetTextureParameterIuivEXT("glGetTextureParameterIuivEXT");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetTextureParameterfv("glGetTextureParameterfv");
Function<void, GLuint, GLenum, GLenum, GLfloat*> Binding::GetTextureParameterfvEXT("glGetTextureParameterfvEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetTextureParameteriv("glGetTextureParameteriv");
Function<void, GLuint, GLenum, GLenum, GLint*> Binding::GetTextureParameterivEXT("glGetTextureParameterivEXT");
Function<GLuint64, GLuint, GLuint> Binding::GetTextureSamplerHandleARB("glGetTextureSamplerHandleARB");
Function<GLuint64, GLuint, GLuint> Binding::GetTextureSamplerHandleNV("glGetTextureSamplerHandleNV");
Function<void, GLuint, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, GLsizei, void*>
    Binding::GetTextureSubImage("glGetTextureSubImage");
Function<void, GLenum, GLuint, GLenum, GLint*> Binding::GetTrackMatrixivNV("glGetTrackMatrixivNV");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLsizei*, GLenum*, GLchar*> Binding::GetTransformFeedbackVarying(
    "glGetTransformFeedbackVarying");
Function<void, GLuint, GLuint, GLsizei, GLsizei*, GLsizei*, GLenum*, GLchar*> Binding::GetTransformFeedbackVaryingEXT(
    "glGetTransformFeedbackVaryingEXT");
Function<void, GLuint, GLuint, GLint*> Binding::GetTransformFeedbackVaryingNV("glGetTransformFeedbackVaryingNV");
Function<void, GLuint, GLenum, GLuint, GLint64*> Binding::GetTransformFeedbacki64_v("glGetTransformFeedbacki64_v");
Function<void, GLuint, GLenum, GLuint, GLint*> Binding::GetTransformFeedbacki_v("glGetTransformFeedbacki_v");
Function<void, GLuint, GLenum, GLint*> Binding::GetTransformFeedbackiv("glGetTransformFeedbackiv");
Function<GLuint, GLuint, const GLchar*> Binding::GetUniformBlockIndex("glGetUniformBlockIndex");
Function<GLint, GLuint, GLint> Binding::GetUniformBufferSizeEXT("glGetUniformBufferSizeEXT");
Function<void, GLuint, GLsizei, const GLchar* const*, GLuint*> Binding::GetUniformIndices("glGetUniformIndices");
Function<GLint, GLuint, const GLchar*> Binding::GetUniformLocation("glGetUniformLocation");
Function<GLint, GLhandleARB, const GLcharARB*> Binding::GetUniformLocationARB("glGetUniformLocationARB");
Function<GLintptr, GLuint, GLint> Binding::GetUniformOffsetEXT("glGetUniformOffsetEXT");
Function<void, GLenum, GLint, GLuint*> Binding::GetUniformSubroutineuiv("glGetUniformSubroutineuiv");
Function<void, GLuint, GLint, GLdouble*> Binding::GetUniformdv("glGetUniformdv");
Function<void, GLuint, GLint, GLfloat*> Binding::GetUniformfv("glGetUniformfv");
Function<void, GLhandleARB, GLint, GLfloat*> Binding::GetUniformfvARB("glGetUniformfvARB");
Function<void, GLuint, GLint, GLint64*> Binding::GetUniformi64vARB("glGetUniformi64vARB");
Function<void, GLuint, GLint, GLint64EXT*> Binding::GetUniformi64vNV("glGetUniformi64vNV");
Function<void, GLuint, GLint, GLint*> Binding::GetUniformiv("glGetUniformiv");
Function<void, GLhandleARB, GLint, GLint*> Binding::GetUniformivARB("glGetUniformivARB");
Function<void, GLuint, GLint, GLuint64*> Binding::GetUniformui64vARB("glGetUniformui64vARB");
Function<void, GLuint, GLint, GLuint64EXT*> Binding::GetUniformui64vNV("glGetUniformui64vNV");
Function<void, GLuint, GLint, GLuint*> Binding::GetUniformuiv("glGetUniformuiv");
Function<void, GLuint, GLint, GLuint*> Binding::GetUniformuivEXT("glGetUniformuivEXT");
Function<void, GLenum, GLuint, GLubyte*> Binding::GetUnsignedBytei_vEXT("glGetUnsignedBytei_vEXT");
Function<void, GLenum, GLubyte*> Binding::GetUnsignedBytevEXT("glGetUnsignedBytevEXT");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetVariantArrayObjectfvATI("glGetVariantArrayObjectfvATI");
Function<void, GLuint, GLenum, GLint*> Binding::GetVariantArrayObjectivATI("glGetVariantArrayObjectivATI");
Function<void, GLuint, GLenum, GLboolean*> Binding::GetVariantBooleanvEXT("glGetVariantBooleanvEXT");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetVariantFloatvEXT("glGetVariantFloatvEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetVariantIntegervEXT("glGetVariantIntegervEXT");
Function<void, GLuint, GLenum, void**> Binding::GetVariantPointervEXT("glGetVariantPointervEXT");
Function<GLint, GLuint, const GLchar*> Binding::GetVaryingLocationNV("glGetVaryingLocationNV");
Function<void, GLuint, GLuint, GLenum, GLint64*> Binding::GetVertexArrayIndexed64iv("glGetVertexArrayIndexed64iv");
Function<void, GLuint, GLuint, GLenum, GLint*> Binding::GetVertexArrayIndexediv("glGetVertexArrayIndexediv");
Function<void, GLuint, GLuint, GLenum, GLint*> Binding::GetVertexArrayIntegeri_vEXT("glGetVertexArrayIntegeri_vEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexArrayIntegervEXT("glGetVertexArrayIntegervEXT");
Function<void, GLuint, GLuint, GLenum, void**> Binding::GetVertexArrayPointeri_vEXT("glGetVertexArrayPointeri_vEXT");
Function<void, GLuint, GLenum, void**> Binding::GetVertexArrayPointervEXT("glGetVertexArrayPointervEXT");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexArrayiv("glGetVertexArrayiv");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetVertexAttribArrayObjectfvATI("glGetVertexAttribArrayObjectfvATI");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexAttribArrayObjectivATI("glGetVertexAttribArrayObjectivATI");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexAttribIiv("glGetVertexAttribIiv");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexAttribIivEXT("glGetVertexAttribIivEXT");
Function<void, GLuint, GLenum, GLuint*> Binding::GetVertexAttribIuiv("glGetVertexAttribIuiv");
Function<void, GLuint, GLenum, GLuint*> Binding::GetVertexAttribIuivEXT("glGetVertexAttribIuivEXT");
Function<void, GLuint, GLenum, GLdouble*> Binding::GetVertexAttribLdv("glGetVertexAttribLdv");
Function<void, GLuint, GLenum, GLdouble*> Binding::GetVertexAttribLdvEXT("glGetVertexAttribLdvEXT");
Function<void, GLuint, GLenum, GLint64EXT*> Binding::GetVertexAttribLi64vNV("glGetVertexAttribLi64vNV");
Function<void, GLuint, GLenum, GLuint64EXT*> Binding::GetVertexAttribLui64vARB("glGetVertexAttribLui64vARB");
Function<void, GLuint, GLenum, GLuint64EXT*> Binding::GetVertexAttribLui64vNV("glGetVertexAttribLui64vNV");
Function<void, GLuint, GLenum, void**> Binding::GetVertexAttribPointerv("glGetVertexAttribPointerv");
Function<void, GLuint, GLenum, void**> Binding::GetVertexAttribPointervARB("glGetVertexAttribPointervARB");
Function<void, GLuint, GLenum, void**> Binding::GetVertexAttribPointervNV("glGetVertexAttribPointervNV");
Function<void, GLuint, GLenum, GLdouble*> Binding::GetVertexAttribdv("glGetVertexAttribdv");
Function<void, GLuint, GLenum, GLdouble*> Binding::GetVertexAttribdvARB("glGetVertexAttribdvARB");
Function<void, GLuint, GLenum, GLdouble*> Binding::GetVertexAttribdvNV("glGetVertexAttribdvNV");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetVertexAttribfv("glGetVertexAttribfv");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetVertexAttribfvARB("glGetVertexAttribfvARB");
Function<void, GLuint, GLenum, GLfloat*> Binding::GetVertexAttribfvNV("glGetVertexAttribfvNV");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexAttribiv("glGetVertexAttribiv");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexAttribivARB("glGetVertexAttribivARB");
Function<void, GLuint, GLenum, GLint*> Binding::GetVertexAttribivNV("glGetVertexAttribivNV");
Function<void, GLuint, GLuint, GLenum, GLdouble*> Binding::GetVideoCaptureStreamdvNV("glGetVideoCaptureStreamdvNV");
Function<void, GLuint, GLuint, GLenum, GLfloat*> Binding::GetVideoCaptureStreamfvNV("glGetVideoCaptureStreamfvNV");
Function<void, GLuint, GLuint, GLenum, GLint*> Binding::GetVideoCaptureStreamivNV("glGetVideoCaptureStreamivNV");
Function<void, GLuint, GLenum, GLint*> Binding::GetVideoCaptureivNV("glGetVideoCaptureivNV");
Function<void, GLuint, GLenum, GLint64EXT*> Binding::GetVideoi64vNV("glGetVideoi64vNV");
Function<void, GLuint, GLenum, GLint*> Binding::GetVideoivNV("glGetVideoivNV");
Function<void, GLuint, GLenum, GLuint64EXT*> Binding::GetVideoui64vNV("glGetVideoui64vNV");
Function<void, GLuint, GLenum, GLuint*> Binding::GetVideouivNV("glGetVideouivNV");
Function<GLVULKANPROCNV, const GLchar*> Binding::GetVkProcAddrNV("glGetVkProcAddrNV");
Function<void, GLenum, GLenum, GLenum, GLsizei, void*> Binding::GetnColorTable("glGetnColorTable");
Function<void, GLenum, GLenum, GLenum, GLsizei, void*> Binding::GetnColorTableARB("glGetnColorTableARB");
Function<void, GLenum, GLint, GLsizei, void*> Binding::GetnCompressedTexImage("glGetnCompressedTexImage");
Function<void, GLenum, GLint, GLsizei, void*> Binding::GetnCompressedTexImageARB("glGetnCompressedTexImageARB");
Function<void, GLenum, GLenum, GLenum, GLsizei, void*> Binding::GetnConvolutionFilter("glGetnConvolutionFilter");
Function<void, GLenum, GLenum, GLenum, GLsizei, void*> Binding::GetnConvolutionFilterARB("glGetnConvolutionFilterARB");
Function<void, GLenum, GLboolean, GLenum, GLenum, GLsizei, void*> Binding::GetnHistogram("glGetnHistogram");
Function<void, GLenum, GLboolean, GLenum, GLenum, GLsizei, void*> Binding::GetnHistogramARB("glGetnHistogramARB");
Function<void, GLenum, GLenum, GLsizei, GLdouble*> Binding::GetnMapdv("glGetnMapdv");
Function<void, GLenum, GLenum, GLsizei, GLdouble*> Binding::GetnMapdvARB("glGetnMapdvARB");
Function<void, GLenum, GLenum, GLsizei, GLfloat*> Binding::GetnMapfv("glGetnMapfv");
Function<void, GLenum, GLenum, GLsizei, GLfloat*> Binding::GetnMapfvARB("glGetnMapfvARB");
Function<void, GLenum, GLenum, GLsizei, GLint*> Binding::GetnMapiv("glGetnMapiv");
Function<void, GLenum, GLenum, GLsizei, GLint*> Binding::GetnMapivARB("glGetnMapivARB");
Function<void, GLenum, GLboolean, GLenum, GLenum, GLsizei, void*> Binding::GetnMinmax("glGetnMinmax");
Function<void, GLenum, GLboolean, GLenum, GLenum, GLsizei, void*> Binding::GetnMinmaxARB("glGetnMinmaxARB");
Function<void, GLenum, GLsizei, GLfloat*> Binding::GetnPixelMapfv("glGetnPixelMapfv");
Function<void, GLenum, GLsizei, GLfloat*> Binding::GetnPixelMapfvARB("glGetnPixelMapfvARB");
Function<void, GLenum, GLsizei, GLuint*> Binding::GetnPixelMapuiv("glGetnPixelMapuiv");
Function<void, GLenum, GLsizei, GLuint*> Binding::GetnPixelMapuivARB("glGetnPixelMapuivARB");
Function<void, GLenum, GLsizei, GLushort*> Binding::GetnPixelMapusv("glGetnPixelMapusv");
Function<void, GLenum, GLsizei, GLushort*> Binding::GetnPixelMapusvARB("glGetnPixelMapusvARB");
Function<void, GLsizei, GLubyte*> Binding::GetnPolygonStipple("glGetnPolygonStipple");
Function<void, GLsizei, GLubyte*> Binding::GetnPolygonStippleARB("glGetnPolygonStippleARB");
Function<void, GLenum, GLenum, GLenum, GLsizei, void*, GLsizei, void*, void*> Binding::GetnSeparableFilter(
    "glGetnSeparableFilter");
Function<void, GLenum, GLenum, GLenum, GLsizei, void*, GLsizei, void*, void*> Binding::GetnSeparableFilterARB(
    "glGetnSeparableFilterARB");
Function<void, GLenum, GLint, GLenum, GLenum, GLsizei, void*> Binding::GetnTexImage("glGetnTexImage");
Function<void, GLenum, GLint, GLenum, GLenum, GLsizei, void*> Binding::GetnTexImageARB("glGetnTexImageARB");
Function<void, GLuint, GLint, GLsizei, GLdouble*> Binding::GetnUniformdv("glGetnUniformdv");
Function<void, GLuint, GLint, GLsizei, GLdouble*> Binding::GetnUniformdvARB("glGetnUniformdvARB");
Function<void, GLuint, GLint, GLsizei, GLfloat*> Binding::GetnUniformfv("glGetnUniformfv");
Function<void, GLuint, GLint, GLsizei, GLfloat*> Binding::GetnUniformfvARB("glGetnUniformfvARB");
Function<void, GLuint, GLint, GLsizei, GLint64*> Binding::GetnUniformi64vARB("glGetnUniformi64vARB");
Function<void, GLuint, GLint, GLsizei, GLint*> Binding::GetnUniformiv("glGetnUniformiv");
Function<void, GLuint, GLint, GLsizei, GLint*> Binding::GetnUniformivARB("glGetnUniformivARB");
Function<void, GLuint, GLint, GLsizei, GLuint64*> Binding::GetnUniformui64vARB("glGetnUniformui64vARB");
Function<void, GLuint, GLint, GLsizei, GLuint*> Binding::GetnUniformuiv("glGetnUniformuiv");
Function<void, GLuint, GLint, GLsizei, GLuint*> Binding::GetnUniformuivARB("glGetnUniformuivARB");
Function<void, GLbyte> Binding::GlobalAlphaFactorbSUN("glGlobalAlphaFactorbSUN");
Function<void, GLdouble> Binding::GlobalAlphaFactordSUN("glGlobalAlphaFactordSUN");
Function<void, GLfloat> Binding::GlobalAlphaFactorfSUN("glGlobalAlphaFactorfSUN");
Function<void, GLint> Binding::GlobalAlphaFactoriSUN("glGlobalAlphaFactoriSUN");
Function<void, GLshort> Binding::GlobalAlphaFactorsSUN("glGlobalAlphaFactorsSUN");
Function<void, GLubyte> Binding::GlobalAlphaFactorubSUN("glGlobalAlphaFactorubSUN");
Function<void, GLuint> Binding::GlobalAlphaFactoruiSUN("glGlobalAlphaFactoruiSUN");
Function<void, GLushort> Binding::GlobalAlphaFactorusSUN("glGlobalAlphaFactorusSUN");



}  // namespace glbinding
