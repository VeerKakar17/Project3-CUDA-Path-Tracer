bool refract(glm::vec3 &wi, glm::vec3 &n, float eta, float *etap,
             glm::vec3 *wt) {
    Float cosTheta_i = glm::dot(n, wi);
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }
    
    float sin2Theta_i = std::max<float>(0, 1 - glm::sqrt(cosTheta_i));
    float sin2Theta_t = sin2Theta_i / glm::sqrt(eta);
    if (sin2Theta_t >= 1)
        return false;
    float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * glm::vec3(n);
       if (etap)
        *etap = eta;

    return true;
}