
#include "Binding_pch.h"


using namespace gl;


namespace glbinding
{
Function<void, GLenum, GLfloat> Binding::PNTrianglesfATI("glPNTrianglesfATI");
Function<void, GLenum, GLint> Binding::PNTrianglesiATI("glPNTrianglesiATI");
Function<void, GLuint, GLuint, GLenum> Binding::PassTexCoordATI("glPassTexCoordATI");
Function<void, GLfloat> Binding::PassThrough("glPassThrough");
Function<void, GLfixed> Binding::PassThroughxOES("glPassThroughxOES");
Function<void, GLenum, const GLfloat*> Binding::PatchParameterfv("glPatchParameterfv");
Function<void, GLenum, GLint> Binding::PatchParameteri("glPatchParameteri");
Function<void, GLenum, GLenum, GLenum, const GLfloat*> Binding::PathColorGenNV("glPathColorGenNV");
Function<void, GLuint, GLsizei, const GLubyte*, GLsizei, GLenum, const void*> Binding::PathCommandsNV(
    "glPathCommandsNV");
Function<void, GLuint, GLsizei, GLenum, const void*> Binding::PathCoordsNV("glPathCoordsNV");
Function<void, GLenum> Binding::PathCoverDepthFuncNV("glPathCoverDepthFuncNV");
Function<void, GLuint, GLsizei, const GLfloat*> Binding::PathDashArrayNV("glPathDashArrayNV");
Function<void, GLenum> Binding::PathFogGenNV("glPathFogGenNV");
Function<GLenum, GLuint, GLenum, const void*, PathFontStyle, GLuint, GLsizei, GLuint, GLfloat>
    Binding::PathGlyphIndexArrayNV("glPathGlyphIndexArrayNV");
Function<GLenum, GLenum, const void*, PathFontStyle, GLuint, GLfloat, GLuint_array_2> Binding::PathGlyphIndexRangeNV(
    "glPathGlyphIndexRangeNV");
Function<void, GLuint, GLenum, const void*, PathFontStyle, GLuint, GLsizei, GLenum, GLuint, GLfloat>
    Binding::PathGlyphRangeNV("glPathGlyphRangeNV");
Function<void, GLuint, GLenum, const void*, PathFontStyle, GLsizei, GLenum, const void*, GLenum, GLuint, GLfloat>
    Binding::PathGlyphsNV("glPathGlyphsNV");
Function<GLenum, GLuint, GLenum, GLsizeiptr, const void*, GLsizei, GLuint, GLsizei, GLuint, GLfloat>
    Binding::PathMemoryGlyphIndexArrayNV("glPathMemoryGlyphIndexArrayNV");
Function<void, GLuint, GLenum, GLfloat> Binding::PathParameterfNV("glPathParameterfNV");
Function<void, GLuint, GLenum, const GLfloat*> Binding::PathParameterfvNV("glPathParameterfvNV");
Function<void, GLuint, GLenum, GLint> Binding::PathParameteriNV("glPathParameteriNV");
Function<void, GLuint, GLenum, const GLint*> Binding::PathParameterivNV("glPathParameterivNV");
Function<void, GLfloat, GLfloat> Binding::PathStencilDepthOffsetNV("glPathStencilDepthOffsetNV");
Function<void, GLenum, GLint, GLuint> Binding::PathStencilFuncNV("glPathStencilFuncNV");
Function<void, GLuint, GLenum, GLsizei, const void*> Binding::PathStringNV("glPathStringNV");
Function<void, GLuint, GLsizei, GLsizei, GLsizei, const GLubyte*, GLsizei, GLenum, const void*>
    Binding::PathSubCommandsNV("glPathSubCommandsNV");
Function<void, GLuint, GLsizei, GLsizei, GLenum, const void*> Binding::PathSubCoordsNV("glPathSubCoordsNV");
Function<void, GLenum, GLenum, GLint, const GLfloat*> Binding::PathTexGenNV("glPathTexGenNV");
Function<void> Binding::PauseTransformFeedback("glPauseTransformFeedback");
Function<void> Binding::PauseTransformFeedbackNV("glPauseTransformFeedbackNV");
Function<void, GLenum, GLsizei, const void*> Binding::PixelDataRangeNV("glPixelDataRangeNV");
Function<void, GLenum, GLsizei, const GLfloat*> Binding::PixelMapfv("glPixelMapfv");
Function<void, GLenum, GLsizei, const GLuint*> Binding::PixelMapuiv("glPixelMapuiv");
Function<void, GLenum, GLsizei, const GLushort*> Binding::PixelMapusv("glPixelMapusv");
Function<void, GLenum, GLint, const GLfixed*> Binding::PixelMapx("glPixelMapx");
Function<void, GLenum, GLfloat> Binding::PixelStoref("glPixelStoref");
Function<void, GLenum, GLint> Binding::PixelStorei("glPixelStorei");
Function<void, GLenum, GLfixed> Binding::PixelStorex("glPixelStorex");
Function<void, GLenum, GLfloat> Binding::PixelTexGenParameterfSGIS("glPixelTexGenParameterfSGIS");
Function<void, GLenum, const GLfloat*> Binding::PixelTexGenParameterfvSGIS("glPixelTexGenParameterfvSGIS");
Function<void, GLenum, GLint> Binding::PixelTexGenParameteriSGIS("glPixelTexGenParameteriSGIS");
Function<void, GLenum, const GLint*> Binding::PixelTexGenParameterivSGIS("glPixelTexGenParameterivSGIS");
Function<void, GLenum> Binding::PixelTexGenSGIX("glPixelTexGenSGIX");
Function<void, GLenum, GLfloat> Binding::PixelTransferf("glPixelTransferf");
Function<void, GLenum, GLint> Binding::PixelTransferi("glPixelTransferi");
Function<void, GLenum, GLfixed> Binding::PixelTransferxOES("glPixelTransferxOES");
Function<void, GLenum, GLenum, GLfloat> Binding::PixelTransformParameterfEXT("glPixelTransformParameterfEXT");
Function<void, GLenum, GLenum, const GLfloat*> Binding::PixelTransformParameterfvEXT("glPixelTransformParameterfvEXT");
Function<void, GLenum, GLenum, GLint> Binding::PixelTransformParameteriEXT("glPixelTransformParameteriEXT");
Function<void, GLenum, GLenum, const GLint*> Binding::PixelTransformParameterivEXT("glPixelTransformParameterivEXT");
Function<void, GLfloat, GLfloat> Binding::PixelZoom("glPixelZoom");
Function<void, GLfixed, GLfixed> Binding::PixelZoomxOES("glPixelZoomxOES");
Function<GLboolean, GLuint, GLsizei, GLsizei, GLfloat, GLfloat*, GLfloat*, GLfloat*, GLfloat*>
    Binding::PointAlongPathNV("glPointAlongPathNV");
Function<void, GLenum, GLfloat> Binding::PointParameterf("glPointParameterf");
Function<void, GLenum, GLfloat> Binding::PointParameterfARB("glPointParameterfARB");
Function<void, GLenum, GLfloat> Binding::PointParameterfEXT("glPointParameterfEXT");
Function<void, GLenum, GLfloat> Binding::PointParameterfSGIS("glPointParameterfSGIS");
Function<void, GLenum, const GLfloat*> Binding::PointParameterfv("glPointParameterfv");
Function<void, GLenum, const GLfloat*> Binding::PointParameterfvARB("glPointParameterfvARB");
Function<void, GLenum, const GLfloat*> Binding::PointParameterfvEXT("glPointParameterfvEXT");
Function<void, GLenum, const GLfloat*> Binding::PointParameterfvSGIS("glPointParameterfvSGIS");
Function<void, GLenum, GLint> Binding::PointParameteri("glPointParameteri");
Function<void, GLenum, GLint> Binding::PointParameteriNV("glPointParameteriNV");
Function<void, GLenum, const GLint*> Binding::PointParameteriv("glPointParameteriv");
Function<void, GLenum, const GLint*> Binding::PointParameterivNV("glPointParameterivNV");
Function<void, GLenum, const GLfixed*> Binding::PointParameterxvOES("glPointParameterxvOES");
Function<void, GLfloat> Binding::PointSize("glPointSize");
Function<void, GLfixed> Binding::PointSizexOES("glPointSizexOES");
Function<GLint, GLuint*> Binding::PollAsyncSGIX("glPollAsyncSGIX");
Function<GLint, GLint*> Binding::PollInstrumentsSGIX("glPollInstrumentsSGIX");
Function<void, GLenum, GLenum> Binding::PolygonMode("glPolygonMode");
Function<void, GLfloat, GLfloat> Binding::PolygonOffset("glPolygonOffset");
Function<void, GLfloat, GLfloat, GLfloat> Binding::PolygonOffsetClamp("glPolygonOffsetClamp");
Function<void, GLfloat, GLfloat, GLfloat> Binding::PolygonOffsetClampEXT("glPolygonOffsetClampEXT");
Function<void, GLfloat, GLfloat> Binding::PolygonOffsetEXT("glPolygonOffsetEXT");
Function<void, GLfixed, GLfixed> Binding::PolygonOffsetxOES("glPolygonOffsetxOES");
Function<void, const GLubyte*> Binding::PolygonStipple("glPolygonStipple");
Function<void> Binding::PopAttrib("glPopAttrib");
Function<void> Binding::PopClientAttrib("glPopClientAttrib");
Function<void> Binding::PopDebugGroup("glPopDebugGroup");
Function<void> Binding::PopGroupMarkerEXT("glPopGroupMarkerEXT");
Function<void> Binding::PopMatrix("glPopMatrix");
Function<void> Binding::PopName("glPopName");
Function<void, GLuint, GLuint64EXT, GLuint, GLuint, GLenum, GLenum, GLuint, GLenum, GLuint, GLenum, GLuint, GLenum,
         GLuint>
    Binding::PresentFrameDualFillNV("glPresentFrameDualFillNV");
Function<void, GLuint, GLuint64EXT, GLuint, GLuint, GLenum, GLenum, GLuint, GLuint, GLenum, GLuint, GLuint>
    Binding::PresentFrameKeyedNV("glPresentFrameKeyedNV");
Function<void, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat> Binding::PrimitiveBoundingBoxARB(
    "glPrimitiveBoundingBoxARB");
Function<void, GLuint> Binding::PrimitiveRestartIndex("glPrimitiveRestartIndex");
Function<void, GLuint> Binding::PrimitiveRestartIndexNV("glPrimitiveRestartIndexNV");
Function<void> Binding::PrimitiveRestartNV("glPrimitiveRestartNV");
Function<void, GLsizei, const GLuint*, const GLfloat*> Binding::PrioritizeTextures("glPrioritizeTextures");
Function<void, GLsizei, const GLuint*, const GLclampf*> Binding::PrioritizeTexturesEXT("glPrioritizeTexturesEXT");
Function<void, GLsizei, const GLuint*, const GLfixed*> Binding::PrioritizeTexturesxOES("glPrioritizeTexturesxOES");
Function<void, GLuint, GLenum, const void*, GLsizei> Binding::ProgramBinary("glProgramBinary");
Function<void, GLenum, GLuint, GLuint, GLsizei, const GLint*> Binding::ProgramBufferParametersIivNV(
    "glProgramBufferParametersIivNV");
Function<void, GLenum, GLuint, GLuint, GLsizei, const GLuint*> Binding::ProgramBufferParametersIuivNV(
    "glProgramBufferParametersIuivNV");
Function<void, GLenum, GLuint, GLuint, GLsizei, const GLfloat*> Binding::ProgramBufferParametersfvNV(
    "glProgramBufferParametersfvNV");
Function<void, GLenum, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::ProgramEnvParameter4dARB(
    "glProgramEnvParameter4dARB");
Function<void, GLenum, GLuint, const GLdouble*> Binding::ProgramEnvParameter4dvARB("glProgramEnvParameter4dvARB");
Function<void, GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ProgramEnvParameter4fARB(
    "glProgramEnvParameter4fARB");
Function<void, GLenum, GLuint, const GLfloat*> Binding::ProgramEnvParameter4fvARB("glProgramEnvParameter4fvARB");
Function<void, GLenum, GLuint, GLint, GLint, GLint, GLint> Binding::ProgramEnvParameterI4iNV(
    "glProgramEnvParameterI4iNV");
Function<void, GLenum, GLuint, const GLint*> Binding::ProgramEnvParameterI4ivNV("glProgramEnvParameterI4ivNV");
Function<void, GLenum, GLuint, GLuint, GLuint, GLuint, GLuint> Binding::ProgramEnvParameterI4uiNV(
    "glProgramEnvParameterI4uiNV");
Function<void, GLenum, GLuint, const GLuint*> Binding::ProgramEnvParameterI4uivNV("glProgramEnvParameterI4uivNV");
Function<void, GLenum, GLuint, GLsizei, const GLfloat*> Binding::ProgramEnvParameters4fvEXT(
    "glProgramEnvParameters4fvEXT");
Function<void, GLenum, GLuint, GLsizei, const GLint*> Binding::ProgramEnvParametersI4ivNV(
    "glProgramEnvParametersI4ivNV");
Function<void, GLenum, GLuint, GLsizei, const GLuint*> Binding::ProgramEnvParametersI4uivNV(
    "glProgramEnvParametersI4uivNV");
Function<void, GLenum, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::ProgramLocalParameter4dARB(
    "glProgramLocalParameter4dARB");
Function<void, GLenum, GLuint, const GLdouble*> Binding::ProgramLocalParameter4dvARB("glProgramLocalParameter4dvARB");
Function<void, GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ProgramLocalParameter4fARB(
    "glProgramLocalParameter4fARB");
Function<void, GLenum, GLuint, const GLfloat*> Binding::ProgramLocalParameter4fvARB("glProgramLocalParameter4fvARB");
Function<void, GLenum, GLuint, GLint, GLint, GLint, GLint> Binding::ProgramLocalParameterI4iNV(
    "glProgramLocalParameterI4iNV");
Function<void, GLenum, GLuint, const GLint*> Binding::ProgramLocalParameterI4ivNV("glProgramLocalParameterI4ivNV");
Function<void, GLenum, GLuint, GLuint, GLuint, GLuint, GLuint> Binding::ProgramLocalParameterI4uiNV(
    "glProgramLocalParameterI4uiNV");
Function<void, GLenum, GLuint, const GLuint*> Binding::ProgramLocalParameterI4uivNV("glProgramLocalParameterI4uivNV");
Function<void, GLenum, GLuint, GLsizei, const GLfloat*> Binding::ProgramLocalParameters4fvEXT(
    "glProgramLocalParameters4fvEXT");
Function<void, GLenum, GLuint, GLsizei, const GLint*> Binding::ProgramLocalParametersI4ivNV(
    "glProgramLocalParametersI4ivNV");
Function<void, GLenum, GLuint, GLsizei, const GLuint*> Binding::ProgramLocalParametersI4uivNV(
    "glProgramLocalParametersI4uivNV");
Function<void, GLuint, GLsizei, const GLubyte*, GLdouble, GLdouble, GLdouble, GLdouble>
    Binding::ProgramNamedParameter4dNV("glProgramNamedParameter4dNV");
Function<void, GLuint, GLsizei, const GLubyte*, const GLdouble*> Binding::ProgramNamedParameter4dvNV(
    "glProgramNamedParameter4dvNV");
Function<void, GLuint, GLsizei, const GLubyte*, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ProgramNamedParameter4fNV(
    "glProgramNamedParameter4fNV");
Function<void, GLuint, GLsizei, const GLubyte*, const GLfloat*> Binding::ProgramNamedParameter4fvNV(
    "glProgramNamedParameter4fvNV");
Function<void, GLenum, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::ProgramParameter4dNV(
    "glProgramParameter4dNV");
Function<void, GLenum, GLuint, const GLdouble*> Binding::ProgramParameter4dvNV("glProgramParameter4dvNV");
Function<void, GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ProgramParameter4fNV(
    "glProgramParameter4fNV");
Function<void, GLenum, GLuint, const GLfloat*> Binding::ProgramParameter4fvNV("glProgramParameter4fvNV");
Function<void, GLuint, GLenum, GLint> Binding::ProgramParameteri("glProgramParameteri");
Function<void, GLuint, GLenum, GLint> Binding::ProgramParameteriARB("glProgramParameteriARB");
Function<void, GLuint, GLenum, GLint> Binding::ProgramParameteriEXT("glProgramParameteriEXT");
Function<void, GLenum, GLuint, GLsizei, const GLdouble*> Binding::ProgramParameters4dvNV("glProgramParameters4dvNV");
Function<void, GLenum, GLuint, GLsizei, const GLfloat*> Binding::ProgramParameters4fvNV("glProgramParameters4fvNV");
Function<void, GLuint, GLint, GLenum, GLint, const GLfloat*> Binding::ProgramPathFragmentInputGenNV(
    "glProgramPathFragmentInputGenNV");
Function<void, GLenum, GLenum, GLsizei, const void*> Binding::ProgramStringARB("glProgramStringARB");
Function<void, GLenum, GLsizei, const GLuint*> Binding::ProgramSubroutineParametersuivNV(
    "glProgramSubroutineParametersuivNV");
Function<void, GLuint, GLint, GLdouble> Binding::ProgramUniform1d("glProgramUniform1d");
Function<void, GLuint, GLint, GLdouble> Binding::ProgramUniform1dEXT("glProgramUniform1dEXT");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform1dv("glProgramUniform1dv");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform1dvEXT("glProgramUniform1dvEXT");
Function<void, GLuint, GLint, GLfloat> Binding::ProgramUniform1f("glProgramUniform1f");
Function<void, GLuint, GLint, GLfloat> Binding::ProgramUniform1fEXT("glProgramUniform1fEXT");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform1fv("glProgramUniform1fv");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform1fvEXT("glProgramUniform1fvEXT");
Function<void, GLuint, GLint, GLint> Binding::ProgramUniform1i("glProgramUniform1i");
Function<void, GLuint, GLint, GLint64> Binding::ProgramUniform1i64ARB("glProgramUniform1i64ARB");
Function<void, GLuint, GLint, GLint64EXT> Binding::ProgramUniform1i64NV("glProgramUniform1i64NV");
Function<void, GLuint, GLint, GLsizei, const GLint64*> Binding::ProgramUniform1i64vARB("glProgramUniform1i64vARB");
Function<void, GLuint, GLint, GLsizei, const GLint64EXT*> Binding::ProgramUniform1i64vNV("glProgramUniform1i64vNV");
Function<void, GLuint, GLint, GLint> Binding::ProgramUniform1iEXT("glProgramUniform1iEXT");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform1iv("glProgramUniform1iv");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform1ivEXT("glProgramUniform1ivEXT");
Function<void, GLuint, GLint, GLuint> Binding::ProgramUniform1ui("glProgramUniform1ui");
Function<void, GLuint, GLint, GLuint64> Binding::ProgramUniform1ui64ARB("glProgramUniform1ui64ARB");
Function<void, GLuint, GLint, GLuint64EXT> Binding::ProgramUniform1ui64NV("glProgramUniform1ui64NV");
Function<void, GLuint, GLint, GLsizei, const GLuint64*> Binding::ProgramUniform1ui64vARB("glProgramUniform1ui64vARB");
Function<void, GLuint, GLint, GLsizei, const GLuint64EXT*> Binding::ProgramUniform1ui64vNV("glProgramUniform1ui64vNV");
Function<void, GLuint, GLint, GLuint> Binding::ProgramUniform1uiEXT("glProgramUniform1uiEXT");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform1uiv("glProgramUniform1uiv");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform1uivEXT("glProgramUniform1uivEXT");
Function<void, GLuint, GLint, GLdouble, GLdouble> Binding::ProgramUniform2d("glProgramUniform2d");
Function<void, GLuint, GLint, GLdouble, GLdouble> Binding::ProgramUniform2dEXT("glProgramUniform2dEXT");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform2dv("glProgramUniform2dv");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform2dvEXT("glProgramUniform2dvEXT");
Function<void, GLuint, GLint, GLfloat, GLfloat> Binding::ProgramUniform2f("glProgramUniform2f");
Function<void, GLuint, GLint, GLfloat, GLfloat> Binding::ProgramUniform2fEXT("glProgramUniform2fEXT");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform2fv("glProgramUniform2fv");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform2fvEXT("glProgramUniform2fvEXT");
Function<void, GLuint, GLint, GLint, GLint> Binding::ProgramUniform2i("glProgramUniform2i");
Function<void, GLuint, GLint, GLint64, GLint64> Binding::ProgramUniform2i64ARB("glProgramUniform2i64ARB");
Function<void, GLuint, GLint, GLint64EXT, GLint64EXT> Binding::ProgramUniform2i64NV("glProgramUniform2i64NV");
Function<void, GLuint, GLint, GLsizei, const GLint64*> Binding::ProgramUniform2i64vARB("glProgramUniform2i64vARB");
Function<void, GLuint, GLint, GLsizei, const GLint64EXT*> Binding::ProgramUniform2i64vNV("glProgramUniform2i64vNV");
Function<void, GLuint, GLint, GLint, GLint> Binding::ProgramUniform2iEXT("glProgramUniform2iEXT");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform2iv("glProgramUniform2iv");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform2ivEXT("glProgramUniform2ivEXT");
Function<void, GLuint, GLint, GLuint, GLuint> Binding::ProgramUniform2ui("glProgramUniform2ui");
Function<void, GLuint, GLint, GLuint64, GLuint64> Binding::ProgramUniform2ui64ARB("glProgramUniform2ui64ARB");
Function<void, GLuint, GLint, GLuint64EXT, GLuint64EXT> Binding::ProgramUniform2ui64NV("glProgramUniform2ui64NV");
Function<void, GLuint, GLint, GLsizei, const GLuint64*> Binding::ProgramUniform2ui64vARB("glProgramUniform2ui64vARB");
Function<void, GLuint, GLint, GLsizei, const GLuint64EXT*> Binding::ProgramUniform2ui64vNV("glProgramUniform2ui64vNV");
Function<void, GLuint, GLint, GLuint, GLuint> Binding::ProgramUniform2uiEXT("glProgramUniform2uiEXT");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform2uiv("glProgramUniform2uiv");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform2uivEXT("glProgramUniform2uivEXT");
Function<void, GLuint, GLint, GLdouble, GLdouble, GLdouble> Binding::ProgramUniform3d("glProgramUniform3d");
Function<void, GLuint, GLint, GLdouble, GLdouble, GLdouble> Binding::ProgramUniform3dEXT("glProgramUniform3dEXT");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform3dv("glProgramUniform3dv");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform3dvEXT("glProgramUniform3dvEXT");
Function<void, GLuint, GLint, GLfloat, GLfloat, GLfloat> Binding::ProgramUniform3f("glProgramUniform3f");
Function<void, GLuint, GLint, GLfloat, GLfloat, GLfloat> Binding::ProgramUniform3fEXT("glProgramUniform3fEXT");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform3fv("glProgramUniform3fv");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform3fvEXT("glProgramUniform3fvEXT");
Function<void, GLuint, GLint, GLint, GLint, GLint> Binding::ProgramUniform3i("glProgramUniform3i");
Function<void, GLuint, GLint, GLint64, GLint64, GLint64> Binding::ProgramUniform3i64ARB("glProgramUniform3i64ARB");
Function<void, GLuint, GLint, GLint64EXT, GLint64EXT, GLint64EXT> Binding::ProgramUniform3i64NV(
    "glProgramUniform3i64NV");
Function<void, GLuint, GLint, GLsizei, const GLint64*> Binding::ProgramUniform3i64vARB("glProgramUniform3i64vARB");
Function<void, GLuint, GLint, GLsizei, const GLint64EXT*> Binding::ProgramUniform3i64vNV("glProgramUniform3i64vNV");
Function<void, GLuint, GLint, GLint, GLint, GLint> Binding::ProgramUniform3iEXT("glProgramUniform3iEXT");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform3iv("glProgramUniform3iv");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform3ivEXT("glProgramUniform3ivEXT");
Function<void, GLuint, GLint, GLuint, GLuint, GLuint> Binding::ProgramUniform3ui("glProgramUniform3ui");
Function<void, GLuint, GLint, GLuint64, GLuint64, GLuint64> Binding::ProgramUniform3ui64ARB("glProgramUniform3ui64ARB");
Function<void, GLuint, GLint, GLuint64EXT, GLuint64EXT, GLuint64EXT> Binding::ProgramUniform3ui64NV(
    "glProgramUniform3ui64NV");
Function<void, GLuint, GLint, GLsizei, const GLuint64*> Binding::ProgramUniform3ui64vARB("glProgramUniform3ui64vARB");
Function<void, GLuint, GLint, GLsizei, const GLuint64EXT*> Binding::ProgramUniform3ui64vNV("glProgramUniform3ui64vNV");
Function<void, GLuint, GLint, GLuint, GLuint, GLuint> Binding::ProgramUniform3uiEXT("glProgramUniform3uiEXT");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform3uiv("glProgramUniform3uiv");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform3uivEXT("glProgramUniform3uivEXT");
Function<void, GLuint, GLint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::ProgramUniform4d("glProgramUniform4d");
Function<void, GLuint, GLint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::ProgramUniform4dEXT(
    "glProgramUniform4dEXT");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform4dv("glProgramUniform4dv");
Function<void, GLuint, GLint, GLsizei, const GLdouble*> Binding::ProgramUniform4dvEXT("glProgramUniform4dvEXT");
Function<void, GLuint, GLint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ProgramUniform4f("glProgramUniform4f");
Function<void, GLuint, GLint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ProgramUniform4fEXT("glProgramUniform4fEXT");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform4fv("glProgramUniform4fv");
Function<void, GLuint, GLint, GLsizei, const GLfloat*> Binding::ProgramUniform4fvEXT("glProgramUniform4fvEXT");
Function<void, GLuint, GLint, GLint, GLint, GLint, GLint> Binding::ProgramUniform4i("glProgramUniform4i");
Function<void, GLuint, GLint, GLint64, GLint64, GLint64, GLint64> Binding::ProgramUniform4i64ARB(
    "glProgramUniform4i64ARB");
Function<void, GLuint, GLint, GLint64EXT, GLint64EXT, GLint64EXT, GLint64EXT> Binding::ProgramUniform4i64NV(
    "glProgramUniform4i64NV");
Function<void, GLuint, GLint, GLsizei, const GLint64*> Binding::ProgramUniform4i64vARB("glProgramUniform4i64vARB");
Function<void, GLuint, GLint, GLsizei, const GLint64EXT*> Binding::ProgramUniform4i64vNV("glProgramUniform4i64vNV");
Function<void, GLuint, GLint, GLint, GLint, GLint, GLint> Binding::ProgramUniform4iEXT("glProgramUniform4iEXT");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform4iv("glProgramUniform4iv");
Function<void, GLuint, GLint, GLsizei, const GLint*> Binding::ProgramUniform4ivEXT("glProgramUniform4ivEXT");
Function<void, GLuint, GLint, GLuint, GLuint, GLuint, GLuint> Binding::ProgramUniform4ui("glProgramUniform4ui");
Function<void, GLuint, GLint, GLuint64, GLuint64, GLuint64, GLuint64> Binding::ProgramUniform4ui64ARB(
    "glProgramUniform4ui64ARB");
Function<void, GLuint, GLint, GLuint64EXT, GLuint64EXT, GLuint64EXT, GLuint64EXT> Binding::ProgramUniform4ui64NV(
    "glProgramUniform4ui64NV");
Function<void, GLuint, GLint, GLsizei, const GLuint64*> Binding::ProgramUniform4ui64vARB("glProgramUniform4ui64vARB");
Function<void, GLuint, GLint, GLsizei, const GLuint64EXT*> Binding::ProgramUniform4ui64vNV("glProgramUniform4ui64vNV");
Function<void, GLuint, GLint, GLuint, GLuint, GLuint, GLuint> Binding::ProgramUniform4uiEXT("glProgramUniform4uiEXT");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform4uiv("glProgramUniform4uiv");
Function<void, GLuint, GLint, GLsizei, const GLuint*> Binding::ProgramUniform4uivEXT("glProgramUniform4uivEXT");
Function<void, GLuint, GLint, GLuint64> Binding::ProgramUniformHandleui64ARB("glProgramUniformHandleui64ARB");
Function<void, GLuint, GLint, GLuint64> Binding::ProgramUniformHandleui64NV("glProgramUniformHandleui64NV");
Function<void, GLuint, GLint, GLsizei, const GLuint64*> Binding::ProgramUniformHandleui64vARB(
    "glProgramUniformHandleui64vARB");
Function<void, GLuint, GLint, GLsizei, const GLuint64*> Binding::ProgramUniformHandleui64vNV(
    "glProgramUniformHandleui64vNV");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix2dv(
    "glProgramUniformMatrix2dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix2dvEXT(
    "glProgramUniformMatrix2dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix2fv(
    "glProgramUniformMatrix2fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix2fvEXT(
    "glProgramUniformMatrix2fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix2x3dv(
    "glProgramUniformMatrix2x3dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix2x3dvEXT(
    "glProgramUniformMatrix2x3dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix2x3fv(
    "glProgramUniformMatrix2x3fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix2x3fvEXT(
    "glProgramUniformMatrix2x3fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix2x4dv(
    "glProgramUniformMatrix2x4dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix2x4dvEXT(
    "glProgramUniformMatrix2x4dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix2x4fv(
    "glProgramUniformMatrix2x4fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix2x4fvEXT(
    "glProgramUniformMatrix2x4fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix3dv(
    "glProgramUniformMatrix3dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix3dvEXT(
    "glProgramUniformMatrix3dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix3fv(
    "glProgramUniformMatrix3fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix3fvEXT(
    "glProgramUniformMatrix3fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix3x2dv(
    "glProgramUniformMatrix3x2dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix3x2dvEXT(
    "glProgramUniformMatrix3x2dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix3x2fv(
    "glProgramUniformMatrix3x2fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix3x2fvEXT(
    "glProgramUniformMatrix3x2fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix3x4dv(
    "glProgramUniformMatrix3x4dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix3x4dvEXT(
    "glProgramUniformMatrix3x4dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix3x4fv(
    "glProgramUniformMatrix3x4fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix3x4fvEXT(
    "glProgramUniformMatrix3x4fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix4dv(
    "glProgramUniformMatrix4dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix4dvEXT(
    "glProgramUniformMatrix4dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix4fv(
    "glProgramUniformMatrix4fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix4fvEXT(
    "glProgramUniformMatrix4fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix4x2dv(
    "glProgramUniformMatrix4x2dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix4x2dvEXT(
    "glProgramUniformMatrix4x2dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix4x2fv(
    "glProgramUniformMatrix4x2fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix4x2fvEXT(
    "glProgramUniformMatrix4x2fvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix4x3dv(
    "glProgramUniformMatrix4x3dv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLdouble*> Binding::ProgramUniformMatrix4x3dvEXT(
    "glProgramUniformMatrix4x3dvEXT");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix4x3fv(
    "glProgramUniformMatrix4x3fv");
Function<void, GLuint, GLint, GLsizei, GLboolean, const GLfloat*> Binding::ProgramUniformMatrix4x3fvEXT(
    "glProgramUniformMatrix4x3fvEXT");
Function<void, GLuint, GLint, GLuint64EXT> Binding::ProgramUniformui64NV("glProgramUniformui64NV");
Function<void, GLuint, GLint, GLsizei, const GLuint64EXT*> Binding::ProgramUniformui64vNV("glProgramUniformui64vNV");
Function<void, GLenum, GLint> Binding::ProgramVertexLimitNV("glProgramVertexLimitNV");
Function<void, GLenum> Binding::ProvokingVertex("glProvokingVertex");
Function<void, GLenum> Binding::ProvokingVertexEXT("glProvokingVertexEXT");
Function<void, AttribMask> Binding::PushAttrib("glPushAttrib");
Function<void, ClientAttribMask> Binding::PushClientAttrib("glPushClientAttrib");
Function<void, ClientAttribMask> Binding::PushClientAttribDefaultEXT("glPushClientAttribDefaultEXT");
Function<void, GLenum, GLuint, GLsizei, const GLchar*> Binding::PushDebugGroup("glPushDebugGroup");
Function<void, GLsizei, const GLchar*> Binding::PushGroupMarkerEXT("glPushGroupMarkerEXT");
Function<void> Binding::PushMatrix("glPushMatrix");
Function<void, GLuint> Binding::PushName("glPushName");



}  // namespace glbinding
