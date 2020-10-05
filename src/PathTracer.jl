module PathTracer

export Vect, Ray, Sphere, Plane, Scene

using LinearAlgebra
using UnPack, Setfield, MuladdMacro
using Images, FileIO
using Printf
using Distributions
using Base.Cartesian
using ProgressMeter
using KernelAbstractions

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
@inline randvect(d::Distribution) = Vect(@ntuple 3 i->rand(d))
Base.promote_rule(::Type{Vect{T}}, ::Type{Vect{S}}) where {T,S} = Vect{promote_type(T, S)}
Base.convert(::Type{Vect{T}}, v::Vect{S}) where {T,S} = Vect(@ntuple 3 i->convert(T, v[i]))
Images.RGB(v::Vect) = RGB((@ntuple 3 i->v[i])...)

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
Base.muladd(v1::Vect, v2::Number, v3::Vect) = Vect(@ntuple 3 i -> muladd(v1[i], v2, v3[i]))
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
@inline @muladd extrapolate(ray::Ray, t) = ray.org + t * ray.dir

struct Intersection{T,P,N,M}
    t::T
    p::P
    n::N
    front::Bool
    material::M
end
@inline function set_face_normal(insec::Intersection, r::Ray, outward_norm::Vect)
    @set! insec.front = front = r.dir'outward_norm < 0
    @set! insec.n = front ? outward_norm : -outward_norm
    return insec
end

###
### Material
###
abstract type AbstractMaterial
end
@enum MaterialType begin
    LAMBERTIAN
    METAL
    DIELECTRIC
end
Base.@kwdef struct Material{C,E,F,I} <: AbstractMaterial
    albedo::C = Vect(0, 0, 0.0)
    emission::E = nothing
    fuzz::F = 0.0
    ir::I = 1.0
    type::MaterialType = LAMBERTIAN
end
@inline @muladd function scatter(insec::Intersection, ray::Ray)
    @unpack dir = ray
    @unpack n, p, material, front = insec
    if material.type === LAMBERTIAN
        scatter_dir = n + random_unit_vector()
        attenutation = material.albedo
        visable = true
    elseif material.type === METAL
        scatter_dir = reflect(dir, n) + material.fuzz * random_in_unit_sphere()
        attenutation = material.albedo
        visable = scatter_dir'n > 0
    else # DIELECTRIC
        attenutation = Vect(1, 1, 1.0)
        @unpack ir = material
        ir = front ? inv(ir) : ir
        scatter_dir = refract(dir, n, ir)
        visable = true
    end
    scattered = Ray(p, scatter_dir)
    return visable, scattered, attenutation
end

@muladd reflect(v::Vect, n::Vect) = v - (2*(v'n))*n # assume normalized
@inline @fastmath @muladd function refract(uv::Vect, n::Vect, ir)
    cosθ = min(-(uv'n), 1)
    sinθ = sqrt(1 - cosθ^2)
    cannot_refract = ir * sinθ > 1
    (cannot_refract || reflectance(cosθ, ir) > rand()) && return reflect(uv, n)
    r_perp = ir * (uv + cosθ*n)
    r_para = -sqrt(abs(1 - r_perp'r_perp)) * n
    return r_perp + r_para
end

@inline function reflectance(cosθ, ir)
    r₀ = (1 - ir) / (1 + ir)
    r₀ = r₀^2
    return r₀ + (1 - r₀)*(1 - cosθ)^5
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
@inline @muladd function Base.intersect(sphere::Sphere, ray::Ray)
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
@inline normal(sphere::Sphere, p0) = normalize(p0 - sphere.center)

struct Plane{V,M} <: AbstractShapes
    p::V
    n::V
    material::M
    Plane(p::V, n::V, material::M) where {V,M} = new{V,M}(p, normalize(n), material)
end

@inline @muladd function Base.intersect(plane::Plane, ray::Ray)
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
struct Camera{L,P,V}
    lens_radius::L
    origin::P
    lower_left_corner::P
    horizontal::V
    vertical::V
    u::V
    v::V
    w::V
end
Camera(lens_radius, args...) = Camera(lens_radius, promote(args...)...)
@muladd function Camera(;
                lookfrom = Vect(0, 0, 0.0),
                lookat = Vect(0, 0, -1.0),
                vup = Vect(0, 1, 0.0),
                vfov = 90,
                aspect_ratio = 16/9,
                aperture = 2.0,
                focus_dist = norm(lookfrom - lookat),
               )
    θ = deg2rad(vfov)
    h = tan(θ/2)
    viewport_height = 2h
    viewport_width = aspect_ratio * viewport_height

    w = normalize(lookfrom - lookat)
    u = normalize(vup × w)
    v = w × u

    origin = lookfrom
    horizontal = focus_dist * viewport_width * u
    vertical = focus_dist * viewport_height * v
    lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w
    lens_radius = aperture / 2
    Camera(lens_radius, origin, lower_left_corner, horizontal, vertical, u, v, w)
end
@muladd function Ray(camera::Camera, s, t)
    @unpack lens_radius, origin, lower_left_corner, horizontal, vertical, u, v, w = camera
    rd = lens_radius * random_in_unit_disk()
    offset = u * rd.x + v * rd.y
    org = origin + offset
    Ray(org, lower_left_corner + s * horizontal + t * vertical - org)
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
@inline function random_in_unit_sphere()
    while true
        p = randvect(Uniform(-1.0, 1.0))
        p'p < 1 && return p
    end
end

@inline random_in_hemisphere(n) = (v = random_in_unit_sphere(); dot(n, v) > 0 ? v : -v)

@inline function random_unit_vector()
    φ = rand(Uniform(0, 2pi))
    z = rand(Uniform(-1, 1.0))
    r = sqrt(1.0 - z^2)
    return Vect(cos(φ)*r, sin(φ)*r, z)
end

@inline function random_in_unit_disk()
    while true
        p = Vect(rand(Uniform(-1, 1.0)), rand(Uniform(-1, 1.0)), 0.0)
        p'p < 1 && return p
    end
end

###
### Scene
###
struct Scene{I,S,P}
    img::I
    spheres::S
    planes::P
    spp::Int # smaples per pixel
end

@inline function Base.intersect(scene::Scene, ray::Ray)
    insec = Intersection(Inf, zero(ray.org), zero(ray.dir), false, first(scene.spheres).material)
    hit = false
    for o in scene.spheres
        t = intersect(o, ray)
        if 1.0e-6 < t < insec.t
            @set! insec.t = t
            @set! insec.p = p = extrapolate(ray, t)
            @set! insec.n = n = normal(o, p)
            insec = set_face_normal(insec, ray, n)
            @set! insec.material = o.material
            hit = true
        end
    end
    return hit, insec
end

###
### Ray tracing
###
@muladd function ray_color(scene::Scene, r::Ray, depth::Int)
    black = Vect(0.0, 0.0, 0.0)
    depth <= 0 && return black
    hit, insec = intersect(scene, r)
    if hit
        visable, scattered, attenutation = scatter(insec, r)
        if visable
            color = ray_color(scene, scattered, depth-1)
            return color * attenutation
        else
            return black
        end
    end
    t = 0.5 * (r.dir.y + 1)
    (1.0-t)*Vect(1.0, 1.0, 1.0) + t*Vect(0.5, 0.7, 1.0)
end

#=
@muladd function raytrace!(scene::Scene, camera::Camera, depth::Int; verbose=true)
    @unpack img, spp = scene
    @unpack origin, lower_left_corner, horizontal, vertical = camera
    he, wi = size(img)
    Threads.@threads for w in 0:wi-1
        for h in 0:he-1
            vcolor = Vect(0.0, 0.0, 0.0)
            for _ in 1:spp
                # anti-aliasing
                u = (w + rand()) / (wi - 1)
                v = (h + rand()) / (he - 1)
                ray = Ray(camera, u, v)
                vcolor += ray_color(scene, ray, depth)
            end
            # gamma correction
            scale = inv(spp)
            color = RGB((@ntuple 3 i->sqrt(scale * vcolor[i]))...)
            img[he-h, w+1] = color
        end
    end
    return scene
end
=#

@kernel function raytrace_kernel!(scene::Scene, camera::Camera, depth::Int; verbose=true)
    h, w = @index(Global, NTuple)
    w -= 1
    h -= 1
    @unpack img, spp = scene
    @unpack origin, lower_left_corner, horizontal, vertical = camera
    he, wi = size(img)
    vcolor = Vect(0.0, 0.0, 0.0)
    for _ in 1:spp
        # anti-aliasing
        u = (w + rand()) / (wi - 1)
        v = (h + rand()) / (he - 1)
        ray = Ray(camera, u, v)
        vcolor += ray_color(scene, ray, depth)
    end
    # gamma correction
    scale = inv(spp)
    color = RGB((@ntuple 3 i->sqrt(scale * vcolor[i]))...)
    img[he-h, w+1] = color
end

function make_scene(;image_height = 1200, aspect_ratio = 3/2, spp=5)
    image_width = floor(Int, image_height * aspect_ratio)
    img = zeros(RGB{Float64}, image_height, image_width)
    material_ground = Material(albedo=Vect(0.5, 0.5, 0.5), type=LAMBERTIAN)
    s = Sphere(Vect(0.0, -1000.0, 0.0), 1000.0, material_ground)
    spheres = [s]
    for a in -11:11, b in -11:11
        choose_mat = rand()
        center = Vect(a + 0.9*rand(), 0.2, b + 0.9*rand())
        if norm(center - Vect(4, 0.2, 0)) > 0.9
            if choose_mat < 0.8
                albedo = randvect(Uniform(0, 1.0)) * randvect(Uniform(0, 1.0))
                sphere_material = Material(albedo=albedo, type=LAMBERTIAN)
            elseif choose_mat < 0.95
                albedo = randvect(Uniform(0, 1.0))
                fuzz = rand(Uniform(0, 0.5))
                sphere_material = Material(albedo=albedo, fuzz=fuzz, type=METAL)
            else
                sphere_material = Material(ir=1.5, type=DIELECTRIC)
            end
            push!(spheres, Sphere(center, 0.2, sphere_material))
        end
    end

    material1 = Material(ir=1.5, type=DIELECTRIC)
    push!(spheres, Sphere(Vect(0.0, 1, 0), 1.0, material1))

    material2 = Material(albedo=Vect(0.4, 0.2, 0.1), type=LAMBERTIAN)
    push!(spheres, Sphere(Vect(-4.0, 1, 0), 1.0, material2))

    material3 = Material(albedo=Vect(0.7, 0.6, 0.5), fuzz=0.0, type=METAL)
    push!(spheres, Sphere(Vect(4.0, 1, 0), 1.0, material3))
    Scene(img, spheres, nothing, spp)
end

make_camera(;aspect_ratio=3/2) = Camera(;
        lookfrom = Vect(13,  2,  3.0),
        lookat   = Vect(0,   0,  0.0),
        vup      = Vect(0,   1,  0.0),
        vfov     = 20,
        aspect_ratio = aspect_ratio,
        aperture = 0.1,
        focus_dist = 10.0,
    )

function main(;
                verbose = true,
                image_height = 1200,
                aspect_ratio = 4/3,
                spp::Int = 5,
                depth::Int = 5,
             )
    scene = make_scene(image_height=image_height, aspect_ratio=aspect_ratio, spp=spp)

    camera = make_camera(aspect_ratio=aspect_ratio)

    raytrace! = raytrace_kernel!(CPU(), 8)
    event = raytrace!(scene, camera, depth; ndrange=size(scene.img))
    wait(event)
    save(File(format"PNG", "ray.png"), scene.img)
end

export main, make_scene, make_camera, raytrace!

end # module
