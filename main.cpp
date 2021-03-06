#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <future>

#define M_PI 3.14159265358979323846
#define INF_DIST 1E20

const int GSampleTimes = 40;
const int GTracingDepth = 5;
const int GProcessNum = 6;

struct Vector3D
{
    float x, y, z;
    explicit Vector3D(float val = 0) : x(val), y(val), z(val) {}
    Vector3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    Vector3D& operator+=(const Vector3D& other) { x += other.x, y += other.y, z += other.z; return *this; }
    Vector3D& operator/=(const float val) { x /= val, y /= val, z /= val; return *this; }
    friend Vector3D operator+(const Vector3D& lhs, const Vector3D& rhs) { return Vector3D(lhs) += rhs; }
    friend Vector3D operator-(const Vector3D& lhs, const Vector3D& rhs) { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; }
    friend Vector3D operator*(const Vector3D& lhs, const Vector3D& rhs) { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; }
    friend Vector3D operator*(const Vector3D& lhs, const float rhs) { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs }; }
    friend Vector3D operator*(const float lhs, const Vector3D& rhs) { return rhs * lhs; }
    friend Vector3D operator/(const Vector3D& lhs, const float rhs) { return Vector3D(lhs) /= rhs; }
    friend bool operator==(const Vector3D& lhs, const Vector3D& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; }
    Vector3D operator-() const { return { -x, -y, -z }; }
    float Length() const { return sqrt(x * x + y * y + z * z); }
    Vector3D Norm() const { return *this / Length(); }
    float Dot(const Vector3D& b) const { return x * b.x + y * b.y + z * b.z; }
    Vector3D Cross(const Vector3D& b) const { return { y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x }; }
};

static Vector3D GCameraPosition{ 278, 273, -600 };
static Vector3D GLightColor(16.0f);

struct Material
{
    Vector3D diffReflctance;
    Vector3D emission;
    Material(const Vector3D& inDiffRefl, const Vector3D& inEmission = Vector3D(0)) : diffReflctance(inDiffRefl), emission(inEmission) {}
};

static Material GDefaultSurface(Vector3D(0.74f, 0.74f, 0.5f));
static Material GGreenSurface(Vector3D(0.05f, 0.4f, 0.06f));
static Material GRedSurface(Vector3D(0.651f, 0.05f, 0.06f));

struct Rectangle
{
    Vector3D a, b, c, d;
    Material mat;
    Vector3D ab, cd, bc, da, normal;
    void CacheVector();
};

void Rectangle::CacheVector()
{
    ab = b - a;
    cd = d - c;
    bc = c - b;
    da = a - d;
    normal = ab.Cross(bc).Norm();
}

Rectangle surfaces[] =
{
    // Floor
    {{553, 0, 0}, {0, 0, 0}, {0,0,559.2}, {553,0,559.2}, GDefaultSurface},
    // Light
    {{343,548.8,227},{343,548.8,332},{213,548.8,332},{213,548.8,227}, { Vector3D(0), GLightColor }},
    // Ceiling
    {{556, 548.8, 0}, {556,548.8,559.2}, {0, 548.8, 559.2}, {0, 548.8, 0}, GDefaultSurface},
    // Back wall
    {{553, 0, 559.2}, {0,0,559.2}, {0,548.8,559.2}, {556,548.8,559.2}, GDefaultSurface},
    // Right wall
    {{0,0,559.2}, Vector3D(0), {0, 548.8,0}, {0,548.8,559.2}, GGreenSurface},
    // Left wall
    {{553, 0,0}, {553, 0, 559.2}, {556, 548.8, 559.2}, {556, 548.8, 0}, GRedSurface},
    // Short block
    {{130, 165, 65},{82, 165, 225},{240, 165, 272},{290,165,114}, GDefaultSurface},
    {{290,0,114},{290,165,114},{240,165,272},{240,0,272}, GDefaultSurface},
    {{130,0,65},{130,165,65},{290,165,114},{290,0,114},GDefaultSurface},
    {{82,0,225},{82,165,225},{130,165,65},{130,0,65},GDefaultSurface},
    {{240,0,272},{240,165,272},{82,165,225},{82,0,225},GDefaultSurface},
    // Tall block
    {{423,330,247},{265,330,296},{314,330,456},{472,330,406},GDefaultSurface},
    {{423,0,247},{423,330,247},{472,330,406},{472,0,406}, GDefaultSurface},
    {{472,0,406},{472,330,406},{314,330,456},{314,0,456},GDefaultSurface},
    {{314,0,456},{314,330,456},{265,330,296},{265,0,296},GDefaultSurface},
    {{265,0,296},{265,330,296},{423,330,247},{423,0,247},GDefaultSurface}
};

static Vector3D GLightPoint(278, 548.8f, 279.5f);
static const int GLightIndex = 1;

struct Ray 
{
    Vector3D position;
    Vector3D direction;
};

bool Intersect(const Ray& ray, int& id, Vector3D& outHitPoint, Vector3D& outNormal)
{
    float nearestDist = INF_DIST;
    bool bHit = false;

    int n = sizeof(surfaces) / sizeof(Rectangle);
    for (int i = 0; i < n; ++i)
    {
        const Rectangle& surface = surfaces[i];
        const Vector3D& surfaceNormal = surface.normal;
        float cosForIncidentAngel = surfaceNormal.Dot(ray.direction);
        if (cosForIncidentAngel == 0)
            continue;

        float t = surfaceNormal.Dot(surface.a - ray.position) / cosForIncidentAngel;

        // avoid a ray intersect with the surface where its position lays on
        if (t <= 0.01f) continue;

        Vector3D hitPoint = ray.position + ray.direction * t;

        Vector3D a2hitpoint = hitPoint - surface.a;
        Vector3D c2hitpoint = hitPoint - surface.c;
        Vector3D d2hitpoint = hitPoint - surface.d;
        Vector3D b2hitpoint = hitPoint - surface.b;

        float testA = surface.ab.Cross(a2hitpoint).Dot(surface.cd.Cross(c2hitpoint));
        float testB = surface.da.Cross(d2hitpoint).Dot(surface.bc.Cross(b2hitpoint));
        if (testA >= 0 && testB >= 0)
        {
            bHit = true;
            if (t + 0.01f < nearestDist)
            {
                nearestDist = t;
                id = i;
                outHitPoint = hitPoint;
                outNormal = cosForIncidentAngel > 0 ? -surfaceNormal : surfaceNormal;
            }
        }
    }
    return bHit;
}

float GetRandomNumber(float lo, float hi)
{
    return lo + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (hi - lo)));
}

Vector3D random_unit_vector_in_hemisphere(const Vector3D& normal)
{
    static std::default_random_engine gen{ 0xBF23 };
    // http://corysimon.github.io/articles/uniformdistn-on-sphere
    static std::normal_distribution<float> dis(0, 1);

    Vector3D candidate(0);
    while (true)
    {
        candidate.x = dis(gen);
        candidate.y = dis(gen);
        candidate.z = dis(gen);
        if (candidate.Length() < 0.001f || candidate.Dot(normal) <= 0) continue;
        return candidate.Norm();
    }
}

Vector3D IndirectLightTrace(const Ray& ray, int depth)
{
    static const float p = 1.0f / (2 * M_PI);
    if (depth > GTracingDepth) return Vector3D(0);

    int hitId = -1;
    Vector3D hitPoint, hitNormal;

    if (!Intersect(ray, hitId, hitPoint, hitNormal)) return Vector3D(0);

    const Rectangle& hitSurface = surfaces[hitId];
    Vector3D emittance = hitSurface.mat.emission;
    if (hitId == GLightIndex)
        return emittance;

    Vector3D BRDF = hitSurface.mat.diffReflctance / M_PI;
    Vector3D incoming;

    Ray newRay;
    newRay.position = hitPoint;
    newRay.direction = random_unit_vector_in_hemisphere(hitNormal);
    incoming = IndirectLightTrace(newRay, depth + 1);
    float cos_theta = newRay.direction.Dot(hitNormal);
    return emittance + BRDF * incoming * cos_theta / p;
}

Vector3D DirectLightTrace(const Ray& incidentRay)
{
    int hitId = -1;
    Vector3D hitPoint, hitNormal;

    if(!Intersect(incidentRay, hitId, hitPoint, hitNormal)) 
        return Vector3D(0);

    const Rectangle& hitSurface = surfaces[hitId];
    Vector3D emittance = hitSurface.mat.emission;

    if (hitId == GLightIndex) return emittance;

    Ray newRay;
    newRay.position = hitPoint;

    {
        Vector3D _hitNormal;
        newRay.direction = (GLightPoint - hitPoint).Norm();
        float cosTheta = newRay.direction.Dot(hitNormal);
        if (Intersect(newRay, hitId, hitPoint, _hitNormal) && hitId == GLightIndex)
            return emittance + hitSurface.mat.diffReflctance * GLightColor * cosTheta;
    }
    return Vector3D(0);
}

Vector3D GammaCorrect(const Vector3D& input)
{
    static float gamma = 1.0f / 2.2f;
    Vector3D color(pow(input.x, gamma), pow(input.y, gamma), pow(input.z, gamma));
    if (color.x > 1.0f) color.x = 1.0f;
    if (color.y > 1.0f) color.y = 1.0f;
    if (color.z > 1.0f) color.z = 1.0f;
    return color;
}


int main(int argc, char* argv[])
{
    clock_t start = clock();

    int width = 1024;
    int height = 768;
    float imageAspectRatio = width / (float)height;

    // just a magic number
    srand(0xBF23);

    std::vector<Vector3D> pixels;
    pixels.resize(width * height);
    for (auto& pixel : pixels) pixel = Vector3D(0);

    for (auto& surface : surfaces) surface.CacheVector();

    auto lambda = [width, height, imageAspectRatio, &pixels](int beginIndex, int endIndex) {
        for (int pixelY = beginIndex; pixelY <= endIndex; ++pixelY)
        {
            const int baseOffset = pixelY * width;
            for (int pixelX = 0; pixelX < width; ++pixelX)
            {
                Vector3D color(0);
                for (int i = 0; i < GSampleTimes; ++i)
                {
                    float px = (1 - 2 * ((pixelX + 0.5f + GetRandomNumber(-0.5f, 0.5f)) / width)) * imageAspectRatio * 0.5135;
                    float py = (1 - 2 * ((pixelY + 0.5f + GetRandomNumber(-0.5f, 0.5f)) / height)) * 0.5135;
                    color += IndirectLightTrace(Ray{ GCameraPosition, Vector3D(px, py, 1).Norm() }, 0);
                }
                color /= GSampleTimes;
                pixels[baseOffset + pixelX] += color;
            }
        }
    };
    
    const int taskStep = height / GProcessNum;
    std::vector<std::future<void>> futures;
    for (int i = 0; i < GProcessNum; ++i) futures.push_back(std::async(lambda, i * taskStep, (i + 1) * taskStep - 1));
    for (auto& fut : futures) fut.wait();

    clock_t end = clock();
    double duration = (end * 1.0f - start) / CLOCKS_PER_SEC;
    printf("\n%f seconds\n", duration);

    for (auto& pixel : pixels) pixel = GammaCorrect(pixel);

    FILE* file = fopen("image.ppm", "w");
    fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; ++i)
        fprintf(file, "%d %d %d ", (int)(pixels[i].x * 255), (int)(pixels[i].y * 255), (int)(pixels[i].z * 255));
    fclose(file);
    return 0;
}