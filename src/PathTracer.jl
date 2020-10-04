module PathTracer

export Vect, Ray, Sphere, Plane, Scene

using LinearAlgebra
using UnPack, Setfield, MuladdMacro
using Images, FileIO
using Printf
using Distributions
using Base.Cartesian

###
### Math
###
struct Vect{T} <: AbstractVector{T}
    x::T
    y::T
    z::T
end
Vect(x, y, z) = Vect(promote(x, y, z)...,)
Base.getindex(v::Vect, i::Int) = getfield(v, i)
Base.size(::Vect) = (3,)
Base.Tuple(v::Vect) = v.x, v.y, v.z
Vect(xx::Tuple) = Vect(xx...,)
Vect(x::Number) = Vect(@ntuple 3 i->x)
vectrand(d::Distribution) where T = Vect(@ntuple 3 i->rand(d))

@muladd LinearAlgebra.dot(v0::Vect, v1::Vect) = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
@muladd LinearAlgebra.cross(v0::Vect, v1::Vect) = Vect(
    v0.y * v1.z - v0.z * v1.y,
    v0.z * v1.x - v0.x * v1.z,
    v0.x * v1.y - v0.y * v1.x,
   )
LinearAlgebra.norm(v::Vect) = sqrt(dot(v, v))
LinearAlgebra.normalize(v::Vect) = v * inv(norm(v))
for op in [:+, :-, :*, :/]
    @eval Base.$op(v0::Vect, v1::Vect) = Vect(@ntuple 3 i -> $op(v0[i], v1[i]))
    @eval Base.$op(v0::Vect, s::Number) = Vect(@ntuple 3 i -> $op(v0[i], s))
    if op !== :/
        @eval Base.$op(s::Number, v0::Vect) = Vect(@ntuple 3 i -> $op(s, v0[i]))
    end
end
Base.:(-)(v::Vect{T}) where T = Vect(@ntuple 3 i -> -v[i])
Base.muladd(v1::Vect, v2::Vect, v3::Vect) = Vect(@ntuple 3 i -> muladd(v1[i], v2[i], v3[i]))
Base.zero(::Type{Vect{T}}) where T = Vect(@ntuple 3 _->zero(T))
Base.zero(::T) where {T<:Vect} = zero(T)

###
### Ray
###
struct Ray{V}
    org::V
    dir::V
    Ray(org::V, dir::V) where V = new{V}(org, normalize(dir))
end
@muladd extrapolate(ray::Ray, t) = ray.org + t * ray.dir

###
### Material
###
struct Material{C,E}
    color::C
    emission::E
    type::Int
end

###
### Shapes
###
abstract type AbstractShapes
end
struct Sphere{V,T,M} <: AbstractShapes
    center::V
    radius::T
    material::M
end
Sphere(c, r) = Sphere(c, r, nothing)
@muladd function Base.intersect(sphere::Sphere, ray::Ray)
    T = eltype(ray.dir)
    rs = ray.org - sphere.center
    B = (2 * rs)'ray.dir
    C = rs'rs - sphere.radius^2
    D = B^2 - 4 * C
    if D > zero(D)
        dist = sqrt(D)
        sol1 = -B + dist
        sol2 = -B - dist
        return sol2 > 1.0e-6 ? sol2/2 :
               sol1 > 1.0e-6 ? sol1/2 : zero(T)
    end
    return zero(T)
end
normal(sphere::Sphere, p0) = normalize(p0 - sphere.center)

struct Plane{V,M} <: AbstractShapes
    p::V
    n::V
    material::M
    Plane(p::V, n::V, material::M) where {V,M} = new{V,M}(p, normalize(n), material)
end
Plane(p, n) = Plane(p, n, nothing)

@muladd function Base.intersect(plane::Plane, ray::Ray)
    T = eltype(ray.dir)
    v = ray.dir'plane.n

    abs(v) < 1.0e-6 && return zero(T)

    iv = -inv(v)
    t = (plane.n'ray.org + plane.p) * iv
    return t < 1.0e-6 ? zero(T) : t
end
normal(plane::Plane, _) = plane.n

###
### Camera and Coordinate transformation
###
struct Camera{P,V}
    origin::P
    lower_left_corner::P
    horizontal::V
    vertical::V
end
Camera(args...) = Camera(promote(args)...)
function Camera(;aspect_ratio=16/9, viewport_height=2, focal_length=1.0)
    viewport_width = aspect_ratio * viewport_height
    origin = Vect(0, 0, 0.0)
    horizontal = Vect(viewport_width, 0, 0.0)
    vertical = Vect(0, viewport_height, 0.0)
    lower_left_corner = origin - horizontal/2 - vertical/2 - Vect(0.0, 0, focal_length)
    Camera(origin, lower_left_corner, horizontal, vertical)
end
function Ray(camera::Camera, u, v)
    @unpack origin, lower_left_corner, horizontal, vertical = camera
    Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin)
end

function xy2xyz((x, y), (w, h))
    fovx = pi / 4
    fovy = (h / w) * fovx
    Vect(
         ((2 * x - w) / w) * tan(fovx),
        -((2 * y - h) / h) * tan(fovy),
        -1.0,
       );
end

function plane2hemishpere(u1, u2)
    r = sqrt(1.0 - u1^2)
    φ = 2 * PI * u2;
    return Vect(cos(φ)*r, sin(φ)*r, u1)
end

###
### Sampling
###
function random_in_unit_sphere()
    while true
        p = vectrand(Uniform(-1.0, 1.0))
        p'p < 1 && return p
    end
end

random_in_hemisphere(n) = (v = random_in_unit_sphere(); dot(n, v) > 0 ? v : -v)

function random_unit_vector()
    φ = rand(Uniform(0, 2pi))
    z = rand(Uniform(-1, 1.0))
    r = sqrt(1.0 - z^2)
    return Vect(cos(φ)*r, sin(φ)*r, z)
end

###
### Scene
###
struct Intersection{T,P,N,O}
    t::T
    p::P
    n::N
    front::Bool
    obj::O
end
@inline function set_face_normal(insec::Intersection, r::Ray, outward_norm::Vect)
    @set! insec.front = front = r.dir'outward_norm < 0
    @set! insec.n = front ? outward_norm : -outward_norm
    return insec
end

struct Scene{I,S,P}
    img::I
    spheres::S
    planes::P
    spp::Int # smaples per pixel
end

function Base.intersect(scene::Scene, ray::Ray)
    insec = Intersection(Inf, zero(ray.org), zero(ray.dir), false, first(scene.spheres))
    for o in scene.spheres
        t = intersect(o, ray)
        if 1.0e-6 < t < insec.t
            @set! insec.t = t
            @set! insec.p = p = extrapolate(ray, t)
            @set! insec.n = n = normal(o, p)
            insec = set_face_normal(insec, ray, n)
            #@set! insec.obj = o
            return true, insec
        end
    end
    return false, insec
end

###
### Ray tracing
###
function ray_color(scene::Scene, r::Ray, depth::Int)
    depth <= 0 && return RGB(0.0, 0.0, 0.0)
    hit, insec = intersect(scene, r)
    if hit
        #target = insec.p + insec.n + random_in_unit_sphere()
        #target = insec.p + insec.n + random_unit_vector()
        target = insec.p + insec.n + random_in_hemisphere(insec.n)
        return 0.5 * ray_color(scene, Ray(insec.p, target - insec.p), depth-1)
    end
    t = 0.5 * (r.dir.y + 1)
    (1.0-t)*RGB(1.0, 1.0, 1.0) + t*RGB(0.5, 0.7, 1.0)
end

@muladd function raytrace!(scene::Scene, camera::Camera, depth::Int; verbose=true)
    @unpack img, spp = scene
    @unpack origin, lower_left_corner, horizontal, vertical = camera
    he, wi = size(img)
    scale = inv(spp)
    for w in 0:wi-1
        verbose && @printf(stderr, "\rScanlines remaining: %5d", wi-1-w); flush(stderr)
        for h in 0:he-1
            pixel_color = RGB(0.0, 0.0, 0.0)
            for _ in 1:spp
                # anti-aliasing
                u = (w + rand()) / (wi - 1)
                v = (h + rand()) / (he - 1)
                ray = Ray(camera, u, v)
                pixel_color += ray_color(scene, ray, depth)
            end
            # gamma correction
            color = RGB((@ntuple 3 i->sqrt(scale * getfield(pixel_color, i)))...)
            img[he-h, w+1] = color
        end
    end
    return scene
end

function main(;
                verbose = true,
                aspect_ratio = 16/9,
                viewport_height = 2,
                focal_length = 1.0,
                spp::Int = 100,
                depth::Int = 50,
             )
    aspect_ratio = 16 / 9
    image_height = 400
    image_width = floor(Int, image_height * aspect_ratio)
    img = zeros(RGB{Float64}, image_height, image_width)
    spheres = (
        Sphere(Vect(0.0, 0, -1), 0.5),
        Sphere(Vect(0.0, -100.5, -1), 100.0),
    )
    camera = Camera(;
        aspect_ratio = aspect_ratio,
        viewport_height = viewport_height,
        focal_length = focal_length,
    )
    scene = Scene(img, spheres, nothing, spp)

    raytrace!(scene, camera, depth; verbose=verbose)
    save(File(format"PNG", "ao.png"), scene.img)
end

export main

end # module
