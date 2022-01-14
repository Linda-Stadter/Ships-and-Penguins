
#include "Binding_pch.h"


using namespace gl;


namespace glbinding
{
Function<void> Binding::VDPAUFiniNV("glVDPAUFiniNV");
Function<void, GLvdpauSurfaceNV, GLenum, GLsizei, GLsizei*, GLint*> Binding::VDPAUGetSurfaceivNV(
    "glVDPAUGetSurfaceivNV");
Function<void, const void*, const void*> Binding::VDPAUInitNV("glVDPAUInitNV");
Function<GLboolean, GLvdpauSurfaceNV> Binding::VDPAUIsSurfaceNV("glVDPAUIsSurfaceNV");
Function<void, GLsizei, const GLvdpauSurfaceNV*> Binding::VDPAUMapSurfacesNV("glVDPAUMapSurfacesNV");
Function<GLvdpauSurfaceNV, const void*, GLenum, GLsizei, const GLuint*> Binding::VDPAURegisterOutputSurfaceNV(
    "glVDPAURegisterOutputSurfaceNV");
Function<GLvdpauSurfaceNV, const void*, GLenum, GLsizei, const GLuint*> Binding::VDPAURegisterVideoSurfaceNV(
    "glVDPAURegisterVideoSurfaceNV");
Function<void, GLvdpauSurfaceNV, GLenum> Binding::VDPAUSurfaceAccessNV("glVDPAUSurfaceAccessNV");
Function<void, GLsizei, const GLvdpauSurfaceNV*> Binding::VDPAUUnmapSurfacesNV("glVDPAUUnmapSurfacesNV");
Function<void, GLvdpauSurfaceNV> Binding::VDPAUUnregisterSurfaceNV("glVDPAUUnregisterSurfaceNV");
Function<void, GLuint> Binding::ValidateProgram("glValidateProgram");
Function<void, GLhandleARB> Binding::ValidateProgramARB("glValidateProgramARB");
Function<void, GLuint> Binding::ValidateProgramPipeline("glValidateProgramPipeline");
Function<void, GLuint, GLenum, GLsizei, GLuint, GLuint> Binding::VariantArrayObjectATI("glVariantArrayObjectATI");
Function<void, GLuint, GLenum, GLuint, const void*> Binding::VariantPointerEXT("glVariantPointerEXT");
Function<void, GLuint, const GLbyte*> Binding::VariantbvEXT("glVariantbvEXT");
Function<void, GLuint, const GLdouble*> Binding::VariantdvEXT("glVariantdvEXT");
Function<void, GLuint, const GLfloat*> Binding::VariantfvEXT("glVariantfvEXT");
Function<void, GLuint, const GLint*> Binding::VariantivEXT("glVariantivEXT");
Function<void, GLuint, const GLshort*> Binding::VariantsvEXT("glVariantsvEXT");
Function<void, GLuint, const GLubyte*> Binding::VariantubvEXT("glVariantubvEXT");
Function<void, GLuint, const GLuint*> Binding::VariantuivEXT("glVariantuivEXT");
Function<void, GLuint, const GLushort*> Binding::VariantusvEXT("glVariantusvEXT");
Function<void, GLbyte, GLbyte> Binding::Vertex2bOES("glVertex2bOES");
Function<void, const GLbyte*> Binding::Vertex2bvOES("glVertex2bvOES");
Function<void, GLdouble, GLdouble> Binding::Vertex2d("glVertex2d");
Function<void, const GLdouble*> Binding::Vertex2dv("glVertex2dv");
Function<void, GLfloat, GLfloat> Binding::Vertex2f("glVertex2f");
Function<void, const GLfloat*> Binding::Vertex2fv("glVertex2fv");
Function<void, GLhalfNV, GLhalfNV> Binding::Vertex2hNV("glVertex2hNV");
Function<void, const GLhalfNV*> Binding::Vertex2hvNV("glVertex2hvNV");
Function<void, GLint, GLint> Binding::Vertex2i("glVertex2i");
Function<void, const GLint*> Binding::Vertex2iv("glVertex2iv");
Function<void, GLshort, GLshort> Binding::Vertex2s("glVertex2s");
Function<void, const GLshort*> Binding::Vertex2sv("glVertex2sv");
Function<void, GLfixed> Binding::Vertex2xOES("glVertex2xOES");
Function<void, const GLfixed*> Binding::Vertex2xvOES("glVertex2xvOES");
Function<void, GLbyte, GLbyte, GLbyte> Binding::Vertex3bOES("glVertex3bOES");
Function<void, const GLbyte*> Binding::Vertex3bvOES("glVertex3bvOES");
Function<void, GLdouble, GLdouble, GLdouble> Binding::Vertex3d("glVertex3d");
Function<void, const GLdouble*> Binding::Vertex3dv("glVertex3dv");
Function<void, GLfloat, GLfloat, GLfloat> Binding::Vertex3f("glVertex3f");
Function<void, const GLfloat*> Binding::Vertex3fv("glVertex3fv");
Function<void, GLhalfNV, GLhalfNV, GLhalfNV> Binding::Vertex3hNV("glVertex3hNV");
Function<void, const GLhalfNV*> Binding::Vertex3hvNV("glVertex3hvNV");
Function<void, GLint, GLint, GLint> Binding::Vertex3i("glVertex3i");
Function<void, const GLint*> Binding::Vertex3iv("glVertex3iv");
Function<void, GLshort, GLshort, GLshort> Binding::Vertex3s("glVertex3s");
Function<void, const GLshort*> Binding::Vertex3sv("glVertex3sv");
Function<void, GLfixed, GLfixed> Binding::Vertex3xOES("glVertex3xOES");
Function<void, const GLfixed*> Binding::Vertex3xvOES("glVertex3xvOES");
Function<void, GLbyte, GLbyte, GLbyte, GLbyte> Binding::Vertex4bOES("glVertex4bOES");
Function<void, const GLbyte*> Binding::Vertex4bvOES("glVertex4bvOES");
Function<void, GLdouble, GLdouble, GLdouble, GLdouble> Binding::Vertex4d("glVertex4d");
Function<void, const GLdouble*> Binding::Vertex4dv("glVertex4dv");
Function<void, GLfloat, GLfloat, GLfloat, GLfloat> Binding::Vertex4f("glVertex4f");
Function<void, const GLfloat*> Binding::Vertex4fv("glVertex4fv");
Function<void, GLhalfNV, GLhalfNV, GLhalfNV, GLhalfNV> Binding::Vertex4hNV("glVertex4hNV");
Function<void, const GLhalfNV*> Binding::Vertex4hvNV("glVertex4hvNV");
Function<void, GLint, GLint, GLint, GLint> Binding::Vertex4i("glVertex4i");
Function<void, const GLint*> Binding::Vertex4iv("glVertex4iv");
Function<void, GLshort, GLshort, GLshort, GLshort> Binding::Vertex4s("glVertex4s");
Function<void, const GLshort*> Binding::Vertex4sv("glVertex4sv");
Function<void, GLfixed, GLfixed, GLfixed> Binding::Vertex4xOES("glVertex4xOES");
Function<void, const GLfixed*> Binding::Vertex4xvOES("glVertex4xvOES");
Function<void, GLuint, GLuint, GLuint> Binding::VertexArrayAttribBinding("glVertexArrayAttribBinding");
Function<void, GLuint, GLuint, GLint, GLenum, GLboolean, GLuint> Binding::VertexArrayAttribFormat(
    "glVertexArrayAttribFormat");
Function<void, GLuint, GLuint, GLint, GLenum, GLuint> Binding::VertexArrayAttribIFormat("glVertexArrayAttribIFormat");
Function<void, GLuint, GLuint, GLint, GLenum, GLuint> Binding::VertexArrayAttribLFormat("glVertexArrayAttribLFormat");
Function<void, GLuint, GLuint, GLuint, GLintptr, GLsizei> Binding::VertexArrayBindVertexBufferEXT(
    "glVertexArrayBindVertexBufferEXT");
Function<void, GLuint, GLuint, GLuint> Binding::VertexArrayBindingDivisor("glVertexArrayBindingDivisor");
Function<void, GLuint, GLuint, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArrayColorOffsetEXT(
    "glVertexArrayColorOffsetEXT");
Function<void, GLuint, GLuint, GLsizei, GLintptr> Binding::VertexArrayEdgeFlagOffsetEXT(
    "glVertexArrayEdgeFlagOffsetEXT");
Function<void, GLuint, GLuint> Binding::VertexArrayElementBuffer("glVertexArrayElementBuffer");
Function<void, GLuint, GLuint, GLenum, GLsizei, GLintptr> Binding::VertexArrayFogCoordOffsetEXT(
    "glVertexArrayFogCoordOffsetEXT");
Function<void, GLuint, GLuint, GLenum, GLsizei, GLintptr> Binding::VertexArrayIndexOffsetEXT(
    "glVertexArrayIndexOffsetEXT");
Function<void, GLuint, GLuint, GLenum, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArrayMultiTexCoordOffsetEXT(
    "glVertexArrayMultiTexCoordOffsetEXT");
Function<void, GLuint, GLuint, GLenum, GLsizei, GLintptr> Binding::VertexArrayNormalOffsetEXT(
    "glVertexArrayNormalOffsetEXT");
Function<void, GLenum, GLint> Binding::VertexArrayParameteriAPPLE("glVertexArrayParameteriAPPLE");
Function<void, GLsizei, void*> Binding::VertexArrayRangeAPPLE("glVertexArrayRangeAPPLE");
Function<void, GLsizei, const void*> Binding::VertexArrayRangeNV("glVertexArrayRangeNV");
Function<void, GLuint, GLuint, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArraySecondaryColorOffsetEXT(
    "glVertexArraySecondaryColorOffsetEXT");
Function<void, GLuint, GLuint, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArrayTexCoordOffsetEXT(
    "glVertexArrayTexCoordOffsetEXT");
Function<void, GLuint, GLuint, GLuint> Binding::VertexArrayVertexAttribBindingEXT(
    "glVertexArrayVertexAttribBindingEXT");
Function<void, GLuint, GLuint, GLuint> Binding::VertexArrayVertexAttribDivisorEXT(
    "glVertexArrayVertexAttribDivisorEXT");
Function<void, GLuint, GLuint, GLint, GLenum, GLboolean, GLuint> Binding::VertexArrayVertexAttribFormatEXT(
    "glVertexArrayVertexAttribFormatEXT");
Function<void, GLuint, GLuint, GLint, GLenum, GLuint> Binding::VertexArrayVertexAttribIFormatEXT(
    "glVertexArrayVertexAttribIFormatEXT");
Function<void, GLuint, GLuint, GLuint, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArrayVertexAttribIOffsetEXT(
    "glVertexArrayVertexAttribIOffsetEXT");
Function<void, GLuint, GLuint, GLint, GLenum, GLuint> Binding::VertexArrayVertexAttribLFormatEXT(
    "glVertexArrayVertexAttribLFormatEXT");
Function<void, GLuint, GLuint, GLuint, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArrayVertexAttribLOffsetEXT(
    "glVertexArrayVertexAttribLOffsetEXT");
Function<void, GLuint, GLuint, GLuint, GLint, GLenum, GLboolean, GLsizei, GLintptr>
    Binding::VertexArrayVertexAttribOffsetEXT("glVertexArrayVertexAttribOffsetEXT");
Function<void, GLuint, GLuint, GLuint> Binding::VertexArrayVertexBindingDivisorEXT(
    "glVertexArrayVertexBindingDivisorEXT");
Function<void, GLuint, GLuint, GLuint, GLintptr, GLsizei> Binding::VertexArrayVertexBuffer("glVertexArrayVertexBuffer");
Function<void, GLuint, GLuint, GLsizei, const GLuint*, const GLintptr*, const GLsizei*>
    Binding::VertexArrayVertexBuffers("glVertexArrayVertexBuffers");
Function<void, GLuint, GLuint, GLint, GLenum, GLsizei, GLintptr> Binding::VertexArrayVertexOffsetEXT(
    "glVertexArrayVertexOffsetEXT");
Function<void, GLuint, GLdouble> Binding::VertexAttrib1d("glVertexAttrib1d");
Function<void, GLuint, GLdouble> Binding::VertexAttrib1dARB("glVertexAttrib1dARB");
Function<void, GLuint, GLdouble> Binding::VertexAttrib1dNV("glVertexAttrib1dNV");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib1dv("glVertexAttrib1dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib1dvARB("glVertexAttrib1dvARB");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib1dvNV("glVertexAttrib1dvNV");
Function<void, GLuint, GLfloat> Binding::VertexAttrib1f("glVertexAttrib1f");
Function<void, GLuint, GLfloat> Binding::VertexAttrib1fARB("glVertexAttrib1fARB");
Function<void, GLuint, GLfloat> Binding::VertexAttrib1fNV("glVertexAttrib1fNV");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib1fv("glVertexAttrib1fv");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib1fvARB("glVertexAttrib1fvARB");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib1fvNV("glVertexAttrib1fvNV");
Function<void, GLuint, GLhalfNV> Binding::VertexAttrib1hNV("glVertexAttrib1hNV");
Function<void, GLuint, const GLhalfNV*> Binding::VertexAttrib1hvNV("glVertexAttrib1hvNV");
Function<void, GLuint, GLshort> Binding::VertexAttrib1s("glVertexAttrib1s");
Function<void, GLuint, GLshort> Binding::VertexAttrib1sARB("glVertexAttrib1sARB");
Function<void, GLuint, GLshort> Binding::VertexAttrib1sNV("glVertexAttrib1sNV");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib1sv("glVertexAttrib1sv");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib1svARB("glVertexAttrib1svARB");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib1svNV("glVertexAttrib1svNV");
Function<void, GLuint, GLdouble, GLdouble> Binding::VertexAttrib2d("glVertexAttrib2d");
Function<void, GLuint, GLdouble, GLdouble> Binding::VertexAttrib2dARB("glVertexAttrib2dARB");
Function<void, GLuint, GLdouble, GLdouble> Binding::VertexAttrib2dNV("glVertexAttrib2dNV");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib2dv("glVertexAttrib2dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib2dvARB("glVertexAttrib2dvARB");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib2dvNV("glVertexAttrib2dvNV");
Function<void, GLuint, GLfloat, GLfloat> Binding::VertexAttrib2f("glVertexAttrib2f");
Function<void, GLuint, GLfloat, GLfloat> Binding::VertexAttrib2fARB("glVertexAttrib2fARB");
Function<void, GLuint, GLfloat, GLfloat> Binding::VertexAttrib2fNV("glVertexAttrib2fNV");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib2fv("glVertexAttrib2fv");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib2fvARB("glVertexAttrib2fvARB");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib2fvNV("glVertexAttrib2fvNV");
Function<void, GLuint, GLhalfNV, GLhalfNV> Binding::VertexAttrib2hNV("glVertexAttrib2hNV");
Function<void, GLuint, const GLhalfNV*> Binding::VertexAttrib2hvNV("glVertexAttrib2hvNV");
Function<void, GLuint, GLshort, GLshort> Binding::VertexAttrib2s("glVertexAttrib2s");
Function<void, GLuint, GLshort, GLshort> Binding::VertexAttrib2sARB("glVertexAttrib2sARB");
Function<void, GLuint, GLshort, GLshort> Binding::VertexAttrib2sNV("glVertexAttrib2sNV");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib2sv("glVertexAttrib2sv");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib2svARB("glVertexAttrib2svARB");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib2svNV("glVertexAttrib2svNV");
Function<void, GLuint, GLdouble, GLdouble, GLdouble> Binding::VertexAttrib3d("glVertexAttrib3d");
Function<void, GLuint, GLdouble, GLdouble, GLdouble> Binding::VertexAttrib3dARB("glVertexAttrib3dARB");
Function<void, GLuint, GLdouble, GLdouble, GLdouble> Binding::VertexAttrib3dNV("glVertexAttrib3dNV");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib3dv("glVertexAttrib3dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib3dvARB("glVertexAttrib3dvARB");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib3dvNV("glVertexAttrib3dvNV");
Function<void, GLuint, GLfloat, GLfloat, GLfloat> Binding::VertexAttrib3f("glVertexAttrib3f");
Function<void, GLuint, GLfloat, GLfloat, GLfloat> Binding::VertexAttrib3fARB("glVertexAttrib3fARB");
Function<void, GLuint, GLfloat, GLfloat, GLfloat> Binding::VertexAttrib3fNV("glVertexAttrib3fNV");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib3fv("glVertexAttrib3fv");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib3fvARB("glVertexAttrib3fvARB");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib3fvNV("glVertexAttrib3fvNV");
Function<void, GLuint, GLhalfNV, GLhalfNV, GLhalfNV> Binding::VertexAttrib3hNV("glVertexAttrib3hNV");
Function<void, GLuint, const GLhalfNV*> Binding::VertexAttrib3hvNV("glVertexAttrib3hvNV");
Function<void, GLuint, GLshort, GLshort, GLshort> Binding::VertexAttrib3s("glVertexAttrib3s");
Function<void, GLuint, GLshort, GLshort, GLshort> Binding::VertexAttrib3sARB("glVertexAttrib3sARB");
Function<void, GLuint, GLshort, GLshort, GLshort> Binding::VertexAttrib3sNV("glVertexAttrib3sNV");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib3sv("glVertexAttrib3sv");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib3svARB("glVertexAttrib3svARB");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib3svNV("glVertexAttrib3svNV");
Function<void, GLuint, const GLbyte*> Binding::VertexAttrib4Nbv("glVertexAttrib4Nbv");
Function<void, GLuint, const GLbyte*> Binding::VertexAttrib4NbvARB("glVertexAttrib4NbvARB");
Function<void, GLuint, const GLint*> Binding::VertexAttrib4Niv("glVertexAttrib4Niv");
Function<void, GLuint, const GLint*> Binding::VertexAttrib4NivARB("glVertexAttrib4NivARB");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib4Nsv("glVertexAttrib4Nsv");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib4NsvARB("glVertexAttrib4NsvARB");
Function<void, GLuint, GLubyte, GLubyte, GLubyte, GLubyte> Binding::VertexAttrib4Nub("glVertexAttrib4Nub");
Function<void, GLuint, GLubyte, GLubyte, GLubyte, GLubyte> Binding::VertexAttrib4NubARB("glVertexAttrib4NubARB");
Function<void, GLuint, const GLubyte*> Binding::VertexAttrib4Nubv("glVertexAttrib4Nubv");
Function<void, GLuint, const GLubyte*> Binding::VertexAttrib4NubvARB("glVertexAttrib4NubvARB");
Function<void, GLuint, const GLuint*> Binding::VertexAttrib4Nuiv("glVertexAttrib4Nuiv");
Function<void, GLuint, const GLuint*> Binding::VertexAttrib4NuivARB("glVertexAttrib4NuivARB");
Function<void, GLuint, const GLushort*> Binding::VertexAttrib4Nusv("glVertexAttrib4Nusv");
Function<void, GLuint, const GLushort*> Binding::VertexAttrib4NusvARB("glVertexAttrib4NusvARB");
Function<void, GLuint, const GLbyte*> Binding::VertexAttrib4bv("glVertexAttrib4bv");
Function<void, GLuint, const GLbyte*> Binding::VertexAttrib4bvARB("glVertexAttrib4bvARB");
Function<void, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::VertexAttrib4d("glVertexAttrib4d");
Function<void, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::VertexAttrib4dARB("glVertexAttrib4dARB");
Function<void, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::VertexAttrib4dNV("glVertexAttrib4dNV");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib4dv("glVertexAttrib4dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib4dvARB("glVertexAttrib4dvARB");
Function<void, GLuint, const GLdouble*> Binding::VertexAttrib4dvNV("glVertexAttrib4dvNV");
Function<void, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::VertexAttrib4f("glVertexAttrib4f");
Function<void, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::VertexAttrib4fARB("glVertexAttrib4fARB");
Function<void, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::VertexAttrib4fNV("glVertexAttrib4fNV");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib4fv("glVertexAttrib4fv");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib4fvARB("glVertexAttrib4fvARB");
Function<void, GLuint, const GLfloat*> Binding::VertexAttrib4fvNV("glVertexAttrib4fvNV");
Function<void, GLuint, GLhalfNV, GLhalfNV, GLhalfNV, GLhalfNV> Binding::VertexAttrib4hNV("glVertexAttrib4hNV");
Function<void, GLuint, const GLhalfNV*> Binding::VertexAttrib4hvNV("glVertexAttrib4hvNV");
Function<void, GLuint, const GLint*> Binding::VertexAttrib4iv("glVertexAttrib4iv");
Function<void, GLuint, const GLint*> Binding::VertexAttrib4ivARB("glVertexAttrib4ivARB");
Function<void, GLuint, GLshort, GLshort, GLshort, GLshort> Binding::VertexAttrib4s("glVertexAttrib4s");
Function<void, GLuint, GLshort, GLshort, GLshort, GLshort> Binding::VertexAttrib4sARB("glVertexAttrib4sARB");
Function<void, GLuint, GLshort, GLshort, GLshort, GLshort> Binding::VertexAttrib4sNV("glVertexAttrib4sNV");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib4sv("glVertexAttrib4sv");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib4svARB("glVertexAttrib4svARB");
Function<void, GLuint, const GLshort*> Binding::VertexAttrib4svNV("glVertexAttrib4svNV");
Function<void, GLuint, GLubyte, GLubyte, GLubyte, GLubyte> Binding::VertexAttrib4ubNV("glVertexAttrib4ubNV");
Function<void, GLuint, const GLubyte*> Binding::VertexAttrib4ubv("glVertexAttrib4ubv");
Function<void, GLuint, const GLubyte*> Binding::VertexAttrib4ubvARB("glVertexAttrib4ubvARB");
Function<void, GLuint, const GLubyte*> Binding::VertexAttrib4ubvNV("glVertexAttrib4ubvNV");
Function<void, GLuint, const GLuint*> Binding::VertexAttrib4uiv("glVertexAttrib4uiv");
Function<void, GLuint, const GLuint*> Binding::VertexAttrib4uivARB("glVertexAttrib4uivARB");
Function<void, GLuint, const GLushort*> Binding::VertexAttrib4usv("glVertexAttrib4usv");
Function<void, GLuint, const GLushort*> Binding::VertexAttrib4usvARB("glVertexAttrib4usvARB");
Function<void, GLuint, GLint, GLenum, GLboolean, GLsizei, GLuint, GLuint> Binding::VertexAttribArrayObjectATI(
    "glVertexAttribArrayObjectATI");
Function<void, GLuint, GLuint> Binding::VertexAttribBinding("glVertexAttribBinding");
Function<void, GLuint, GLuint> Binding::VertexAttribDivisor("glVertexAttribDivisor");
Function<void, GLuint, GLuint> Binding::VertexAttribDivisorARB("glVertexAttribDivisorARB");
Function<void, GLuint, GLint, GLenum, GLboolean, GLuint> Binding::VertexAttribFormat("glVertexAttribFormat");
Function<void, GLuint, GLint, GLenum, GLboolean, GLsizei> Binding::VertexAttribFormatNV("glVertexAttribFormatNV");
Function<void, GLuint, GLint> Binding::VertexAttribI1i("glVertexAttribI1i");
Function<void, GLuint, GLint> Binding::VertexAttribI1iEXT("glVertexAttribI1iEXT");
Function<void, GLuint, const GLint*> Binding::VertexAttribI1iv("glVertexAttribI1iv");
Function<void, GLuint, const GLint*> Binding::VertexAttribI1ivEXT("glVertexAttribI1ivEXT");
Function<void, GLuint, GLuint> Binding::VertexAttribI1ui("glVertexAttribI1ui");
Function<void, GLuint, GLuint> Binding::VertexAttribI1uiEXT("glVertexAttribI1uiEXT");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI1uiv("glVertexAttribI1uiv");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI1uivEXT("glVertexAttribI1uivEXT");
Function<void, GLuint, GLint, GLint> Binding::VertexAttribI2i("glVertexAttribI2i");
Function<void, GLuint, GLint, GLint> Binding::VertexAttribI2iEXT("glVertexAttribI2iEXT");
Function<void, GLuint, const GLint*> Binding::VertexAttribI2iv("glVertexAttribI2iv");
Function<void, GLuint, const GLint*> Binding::VertexAttribI2ivEXT("glVertexAttribI2ivEXT");
Function<void, GLuint, GLuint, GLuint> Binding::VertexAttribI2ui("glVertexAttribI2ui");
Function<void, GLuint, GLuint, GLuint> Binding::VertexAttribI2uiEXT("glVertexAttribI2uiEXT");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI2uiv("glVertexAttribI2uiv");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI2uivEXT("glVertexAttribI2uivEXT");
Function<void, GLuint, GLint, GLint, GLint> Binding::VertexAttribI3i("glVertexAttribI3i");
Function<void, GLuint, GLint, GLint, GLint> Binding::VertexAttribI3iEXT("glVertexAttribI3iEXT");
Function<void, GLuint, const GLint*> Binding::VertexAttribI3iv("glVertexAttribI3iv");
Function<void, GLuint, const GLint*> Binding::VertexAttribI3ivEXT("glVertexAttribI3ivEXT");
Function<void, GLuint, GLuint, GLuint, GLuint> Binding::VertexAttribI3ui("glVertexAttribI3ui");
Function<void, GLuint, GLuint, GLuint, GLuint> Binding::VertexAttribI3uiEXT("glVertexAttribI3uiEXT");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI3uiv("glVertexAttribI3uiv");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI3uivEXT("glVertexAttribI3uivEXT");
Function<void, GLuint, const GLbyte*> Binding::VertexAttribI4bv("glVertexAttribI4bv");
Function<void, GLuint, const GLbyte*> Binding::VertexAttribI4bvEXT("glVertexAttribI4bvEXT");
Function<void, GLuint, GLint, GLint, GLint, GLint> Binding::VertexAttribI4i("glVertexAttribI4i");
Function<void, GLuint, GLint, GLint, GLint, GLint> Binding::VertexAttribI4iEXT("glVertexAttribI4iEXT");
Function<void, GLuint, const GLint*> Binding::VertexAttribI4iv("glVertexAttribI4iv");
Function<void, GLuint, const GLint*> Binding::VertexAttribI4ivEXT("glVertexAttribI4ivEXT");
Function<void, GLuint, const GLshort*> Binding::VertexAttribI4sv("glVertexAttribI4sv");
Function<void, GLuint, const GLshort*> Binding::VertexAttribI4svEXT("glVertexAttribI4svEXT");
Function<void, GLuint, const GLubyte*> Binding::VertexAttribI4ubv("glVertexAttribI4ubv");
Function<void, GLuint, const GLubyte*> Binding::VertexAttribI4ubvEXT("glVertexAttribI4ubvEXT");
Function<void, GLuint, GLuint, GLuint, GLuint, GLuint> Binding::VertexAttribI4ui("glVertexAttribI4ui");
Function<void, GLuint, GLuint, GLuint, GLuint, GLuint> Binding::VertexAttribI4uiEXT("glVertexAttribI4uiEXT");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI4uiv("glVertexAttribI4uiv");
Function<void, GLuint, const GLuint*> Binding::VertexAttribI4uivEXT("glVertexAttribI4uivEXT");
Function<void, GLuint, const GLushort*> Binding::VertexAttribI4usv("glVertexAttribI4usv");
Function<void, GLuint, const GLushort*> Binding::VertexAttribI4usvEXT("glVertexAttribI4usvEXT");
Function<void, GLuint, GLint, GLenum, GLuint> Binding::VertexAttribIFormat("glVertexAttribIFormat");
Function<void, GLuint, GLint, GLenum, GLsizei> Binding::VertexAttribIFormatNV("glVertexAttribIFormatNV");
Function<void, GLuint, GLint, GLenum, GLsizei, const void*> Binding::VertexAttribIPointer("glVertexAttribIPointer");
Function<void, GLuint, GLint, GLenum, GLsizei, const void*> Binding::VertexAttribIPointerEXT(
    "glVertexAttribIPointerEXT");
Function<void, GLuint, GLdouble> Binding::VertexAttribL1d("glVertexAttribL1d");
Function<void, GLuint, GLdouble> Binding::VertexAttribL1dEXT("glVertexAttribL1dEXT");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL1dv("glVertexAttribL1dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL1dvEXT("glVertexAttribL1dvEXT");
Function<void, GLuint, GLint64EXT> Binding::VertexAttribL1i64NV("glVertexAttribL1i64NV");
Function<void, GLuint, const GLint64EXT*> Binding::VertexAttribL1i64vNV("glVertexAttribL1i64vNV");
Function<void, GLuint, GLuint64EXT> Binding::VertexAttribL1ui64ARB("glVertexAttribL1ui64ARB");
Function<void, GLuint, GLuint64EXT> Binding::VertexAttribL1ui64NV("glVertexAttribL1ui64NV");
Function<void, GLuint, const GLuint64EXT*> Binding::VertexAttribL1ui64vARB("glVertexAttribL1ui64vARB");
Function<void, GLuint, const GLuint64EXT*> Binding::VertexAttribL1ui64vNV("glVertexAttribL1ui64vNV");
Function<void, GLuint, GLdouble, GLdouble> Binding::VertexAttribL2d("glVertexAttribL2d");
Function<void, GLuint, GLdouble, GLdouble> Binding::VertexAttribL2dEXT("glVertexAttribL2dEXT");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL2dv("glVertexAttribL2dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL2dvEXT("glVertexAttribL2dvEXT");
Function<void, GLuint, GLint64EXT, GLint64EXT> Binding::VertexAttribL2i64NV("glVertexAttribL2i64NV");
Function<void, GLuint, const GLint64EXT*> Binding::VertexAttribL2i64vNV("glVertexAttribL2i64vNV");
Function<void, GLuint, GLuint64EXT, GLuint64EXT> Binding::VertexAttribL2ui64NV("glVertexAttribL2ui64NV");
Function<void, GLuint, const GLuint64EXT*> Binding::VertexAttribL2ui64vNV("glVertexAttribL2ui64vNV");
Function<void, GLuint, GLdouble, GLdouble, GLdouble> Binding::VertexAttribL3d("glVertexAttribL3d");
Function<void, GLuint, GLdouble, GLdouble, GLdouble> Binding::VertexAttribL3dEXT("glVertexAttribL3dEXT");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL3dv("glVertexAttribL3dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL3dvEXT("glVertexAttribL3dvEXT");
Function<void, GLuint, GLint64EXT, GLint64EXT, GLint64EXT> Binding::VertexAttribL3i64NV("glVertexAttribL3i64NV");
Function<void, GLuint, const GLint64EXT*> Binding::VertexAttribL3i64vNV("glVertexAttribL3i64vNV");
Function<void, GLuint, GLuint64EXT, GLuint64EXT, GLuint64EXT> Binding::VertexAttribL3ui64NV("glVertexAttribL3ui64NV");
Function<void, GLuint, const GLuint64EXT*> Binding::VertexAttribL3ui64vNV("glVertexAttribL3ui64vNV");
Function<void, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::VertexAttribL4d("glVertexAttribL4d");
Function<void, GLuint, GLdouble, GLdouble, GLdouble, GLdouble> Binding::VertexAttribL4dEXT("glVertexAttribL4dEXT");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL4dv("glVertexAttribL4dv");
Function<void, GLuint, const GLdouble*> Binding::VertexAttribL4dvEXT("glVertexAttribL4dvEXT");
Function<void, GLuint, GLint64EXT, GLint64EXT, GLint64EXT, GLint64EXT> Binding::VertexAttribL4i64NV(
    "glVertexAttribL4i64NV");
Function<void, GLuint, const GLint64EXT*> Binding::VertexAttribL4i64vNV("glVertexAttribL4i64vNV");
Function<void, GLuint, GLuint64EXT, GLuint64EXT, GLuint64EXT, GLuint64EXT> Binding::VertexAttribL4ui64NV(
    "glVertexAttribL4ui64NV");
Function<void, GLuint, const GLuint64EXT*> Binding::VertexAttribL4ui64vNV("glVertexAttribL4ui64vNV");
Function<void, GLuint, GLint, GLenum, GLuint> Binding::VertexAttribLFormat("glVertexAttribLFormat");
Function<void, GLuint, GLint, GLenum, GLsizei> Binding::VertexAttribLFormatNV("glVertexAttribLFormatNV");
Function<void, GLuint, GLint, GLenum, GLsizei, const void*> Binding::VertexAttribLPointer("glVertexAttribLPointer");
Function<void, GLuint, GLint, GLenum, GLsizei, const void*> Binding::VertexAttribLPointerEXT(
    "glVertexAttribLPointerEXT");
Function<void, GLuint, GLenum, GLboolean, GLuint> Binding::VertexAttribP1ui("glVertexAttribP1ui");
Function<void, GLuint, GLenum, GLboolean, const GLuint*> Binding::VertexAttribP1uiv("glVertexAttribP1uiv");
Function<void, GLuint, GLenum, GLboolean, GLuint> Binding::VertexAttribP2ui("glVertexAttribP2ui");
Function<void, GLuint, GLenum, GLboolean, const GLuint*> Binding::VertexAttribP2uiv("glVertexAttribP2uiv");
Function<void, GLuint, GLenum, GLboolean, GLuint> Binding::VertexAttribP3ui("glVertexAttribP3ui");
Function<void, GLuint, GLenum, GLboolean, const GLuint*> Binding::VertexAttribP3uiv("glVertexAttribP3uiv");
Function<void, GLuint, GLenum, GLboolean, GLuint> Binding::VertexAttribP4ui("glVertexAttribP4ui");
Function<void, GLuint, GLenum, GLboolean, const GLuint*> Binding::VertexAttribP4uiv("glVertexAttribP4uiv");
Function<void, GLuint, GLenum, GLint> Binding::VertexAttribParameteriAMD("glVertexAttribParameteriAMD");
Function<void, GLuint, GLint, GLenum, GLboolean, GLsizei, const void*> Binding::VertexAttribPointer(
    "glVertexAttribPointer");
Function<void, GLuint, GLint, GLenum, GLboolean, GLsizei, const void*> Binding::VertexAttribPointerARB(
    "glVertexAttribPointerARB");
Function<void, GLuint, GLint, GLenum, GLsizei, const void*> Binding::VertexAttribPointerNV("glVertexAttribPointerNV");
Function<void, GLuint, GLsizei, const GLdouble*> Binding::VertexAttribs1dvNV("glVertexAttribs1dvNV");
Function<void, GLuint, GLsizei, const GLfloat*> Binding::VertexAttribs1fvNV("glVertexAttribs1fvNV");
Function<void, GLuint, GLsizei, const GLhalfNV*> Binding::VertexAttribs1hvNV("glVertexAttribs1hvNV");
Function<void, GLuint, GLsizei, const GLshort*> Binding::VertexAttribs1svNV("glVertexAttribs1svNV");
Function<void, GLuint, GLsizei, const GLdouble*> Binding::VertexAttribs2dvNV("glVertexAttribs2dvNV");
Function<void, GLuint, GLsizei, const GLfloat*> Binding::VertexAttribs2fvNV("glVertexAttribs2fvNV");
Function<void, GLuint, GLsizei, const GLhalfNV*> Binding::VertexAttribs2hvNV("glVertexAttribs2hvNV");
Function<void, GLuint, GLsizei, const GLshort*> Binding::VertexAttribs2svNV("glVertexAttribs2svNV");
Function<void, GLuint, GLsizei, const GLdouble*> Binding::VertexAttribs3dvNV("glVertexAttribs3dvNV");
Function<void, GLuint, GLsizei, const GLfloat*> Binding::VertexAttribs3fvNV("glVertexAttribs3fvNV");
Function<void, GLuint, GLsizei, const GLhalfNV*> Binding::VertexAttribs3hvNV("glVertexAttribs3hvNV");
Function<void, GLuint, GLsizei, const GLshort*> Binding::VertexAttribs3svNV("glVertexAttribs3svNV");
Function<void, GLuint, GLsizei, const GLdouble*> Binding::VertexAttribs4dvNV("glVertexAttribs4dvNV");
Function<void, GLuint, GLsizei, const GLfloat*> Binding::VertexAttribs4fvNV("glVertexAttribs4fvNV");
Function<void, GLuint, GLsizei, const GLhalfNV*> Binding::VertexAttribs4hvNV("glVertexAttribs4hvNV");
Function<void, GLuint, GLsizei, const GLshort*> Binding::VertexAttribs4svNV("glVertexAttribs4svNV");
Function<void, GLuint, GLsizei, const GLubyte*> Binding::VertexAttribs4ubvNV("glVertexAttribs4ubvNV");
Function<void, GLuint, GLuint> Binding::VertexBindingDivisor("glVertexBindingDivisor");
Function<void, GLint> Binding::VertexBlendARB("glVertexBlendARB");
Function<void, GLenum, GLfloat> Binding::VertexBlendEnvfATI("glVertexBlendEnvfATI");
Function<void, GLenum, GLint> Binding::VertexBlendEnviATI("glVertexBlendEnviATI");
Function<void, GLint, GLenum, GLsizei> Binding::VertexFormatNV("glVertexFormatNV");
Function<void, GLenum, GLuint> Binding::VertexP2ui("glVertexP2ui");
Function<void, GLenum, const GLuint*> Binding::VertexP2uiv("glVertexP2uiv");
Function<void, GLenum, GLuint> Binding::VertexP3ui("glVertexP3ui");
Function<void, GLenum, const GLuint*> Binding::VertexP3uiv("glVertexP3uiv");
Function<void, GLenum, GLuint> Binding::VertexP4ui("glVertexP4ui");
Function<void, GLenum, const GLuint*> Binding::VertexP4uiv("glVertexP4uiv");
Function<void, GLint, GLenum, GLsizei, const void*> Binding::VertexPointer("glVertexPointer");
Function<void, GLint, GLenum, GLsizei, GLsizei, const void*> Binding::VertexPointerEXT("glVertexPointerEXT");
Function<void, GLint, GLenum, GLint, const void**, GLint> Binding::VertexPointerListIBM("glVertexPointerListIBM");
Function<void, GLint, GLenum, const void**> Binding::VertexPointervINTEL("glVertexPointervINTEL");
Function<void, GLenum, GLdouble> Binding::VertexStream1dATI("glVertexStream1dATI");
Function<void, GLenum, const GLdouble*> Binding::VertexStream1dvATI("glVertexStream1dvATI");
Function<void, GLenum, GLfloat> Binding::VertexStream1fATI("glVertexStream1fATI");
Function<void, GLenum, const GLfloat*> Binding::VertexStream1fvATI("glVertexStream1fvATI");
Function<void, GLenum, GLint> Binding::VertexStream1iATI("glVertexStream1iATI");
Function<void, GLenum, const GLint*> Binding::VertexStream1ivATI("glVertexStream1ivATI");
Function<void, GLenum, GLshort> Binding::VertexStream1sATI("glVertexStream1sATI");
Function<void, GLenum, const GLshort*> Binding::VertexStream1svATI("glVertexStream1svATI");
Function<void, GLenum, GLdouble, GLdouble> Binding::VertexStream2dATI("glVertexStream2dATI");
Function<void, GLenum, const GLdouble*> Binding::VertexStream2dvATI("glVertexStream2dvATI");
Function<void, GLenum, GLfloat, GLfloat> Binding::VertexStream2fATI("glVertexStream2fATI");
Function<void, GLenum, const GLfloat*> Binding::VertexStream2fvATI("glVertexStream2fvATI");
Function<void, GLenum, GLint, GLint> Binding::VertexStream2iATI("glVertexStream2iATI");
Function<void, GLenum, const GLint*> Binding::VertexStream2ivATI("glVertexStream2ivATI");
Function<void, GLenum, GLshort, GLshort> Binding::VertexStream2sATI("glVertexStream2sATI");
Function<void, GLenum, const GLshort*> Binding::VertexStream2svATI("glVertexStream2svATI");
Function<void, GLenum, GLdouble, GLdouble, GLdouble> Binding::VertexStream3dATI("glVertexStream3dATI");
Function<void, GLenum, const GLdouble*> Binding::VertexStream3dvATI("glVertexStream3dvATI");
Function<void, GLenum, GLfloat, GLfloat, GLfloat> Binding::VertexStream3fATI("glVertexStream3fATI");
Function<void, GLenum, const GLfloat*> Binding::VertexStream3fvATI("glVertexStream3fvATI");
Function<void, GLenum, GLint, GLint, GLint> Binding::VertexStream3iATI("glVertexStream3iATI");
Function<void, GLenum, const GLint*> Binding::VertexStream3ivATI("glVertexStream3ivATI");
Function<void, GLenum, GLshort, GLshort, GLshort> Binding::VertexStream3sATI("glVertexStream3sATI");
Function<void, GLenum, const GLshort*> Binding::VertexStream3svATI("glVertexStream3svATI");
Function<void, GLenum, GLdouble, GLdouble, GLdouble, GLdouble> Binding::VertexStream4dATI("glVertexStream4dATI");
Function<void, GLenum, const GLdouble*> Binding::VertexStream4dvATI("glVertexStream4dvATI");
Function<void, GLenum, GLfloat, GLfloat, GLfloat, GLfloat> Binding::VertexStream4fATI("glVertexStream4fATI");
Function<void, GLenum, const GLfloat*> Binding::VertexStream4fvATI("glVertexStream4fvATI");
Function<void, GLenum, GLint, GLint, GLint, GLint> Binding::VertexStream4iATI("glVertexStream4iATI");
Function<void, GLenum, const GLint*> Binding::VertexStream4ivATI("glVertexStream4ivATI");
Function<void, GLenum, GLshort, GLshort, GLshort, GLshort> Binding::VertexStream4sATI("glVertexStream4sATI");
Function<void, GLenum, const GLshort*> Binding::VertexStream4svATI("glVertexStream4svATI");
Function<void, GLint, GLenum, GLsizei, const void*> Binding::VertexWeightPointerEXT("glVertexWeightPointerEXT");
Function<void, GLfloat> Binding::VertexWeightfEXT("glVertexWeightfEXT");
Function<void, const GLfloat*> Binding::VertexWeightfvEXT("glVertexWeightfvEXT");
Function<void, GLhalfNV> Binding::VertexWeighthNV("glVertexWeighthNV");
Function<void, const GLhalfNV*> Binding::VertexWeighthvNV("glVertexWeighthvNV");
Function<GLenum, GLuint, GLuint*, GLuint64EXT*> Binding::VideoCaptureNV("glVideoCaptureNV");
Function<void, GLuint, GLuint, GLenum, const GLdouble*> Binding::VideoCaptureStreamParameterdvNV(
    "glVideoCaptureStreamParameterdvNV");
Function<void, GLuint, GLuint, GLenum, const GLfloat*> Binding::VideoCaptureStreamParameterfvNV(
    "glVideoCaptureStreamParameterfvNV");
Function<void, GLuint, GLuint, GLenum, const GLint*> Binding::VideoCaptureStreamParameterivNV(
    "glVideoCaptureStreamParameterivNV");
Function<void, GLint, GLint, GLsizei, GLsizei> Binding::Viewport("glViewport");
Function<void, GLuint, GLsizei, const GLfloat*> Binding::ViewportArrayv("glViewportArrayv");
Function<void, GLuint, GLfloat, GLfloat, GLfloat, GLfloat> Binding::ViewportIndexedf("glViewportIndexedf");
Function<void, GLuint, const GLfloat*> Binding::ViewportIndexedfv("glViewportIndexedfv");
Function<void, GLuint, GLfloat, GLfloat> Binding::ViewportPositionWScaleNV("glViewportPositionWScaleNV");
Function<void, GLuint, GLenum, GLenum, GLenum, GLenum> Binding::ViewportSwizzleNV("glViewportSwizzleNV");



}  // namespace glbinding
