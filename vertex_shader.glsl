#version 330 core

layout (location = 0) in vec4 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 transformMatrix;

void main()
{
    gl_Position = transformMatrix * aPos;
    TexCoord = aTexCoord;
}