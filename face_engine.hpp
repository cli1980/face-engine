#pragma once

#include "net.hpp"
#include <string>
#include <map>
#include <vector>

using embedding_map = std::map<string, std::vector<matrix<float,0,1>>>;
class FaceEngine
{
public:
    FaceEngine() = delete;
    
    FaceEngine(const string& landmark /*< path to parameter file of landmark*/,
               const string& fr /* path to parameter file of fr*/);

    virtual ~FaceEngine();

    /* Generate embeddings on given data set */
    virtual bool BuildDataset(const string& dpath /*< path to dataset */,
                              const string& epath = "embeddings" /*< path to embeddings */);

    /* Load embeddings from file system */
    virtual bool LoadEmbeddings(const string& path = "embeddings" /*< path to embeddings */);

    typedef struct
    {
        rectangle bbox;
        string    name;
    } Label;
    
    /* Evalueate and label the faces in a given image */
    virtual bool Evaluate(const matrix<rgb_pixel> &img, std::vector<Label>& labels, 
                           double threshold = 0.6);

private:
    shape_predictor         m_shapePred;
    anet_type               m_frNet;
    frontal_face_detector   m_faceDetector;
    embedding_map           m_faceMap;

protected:
    virtual bool _GenerateEmbeddings(const string& name /*< person name*/,
                                     const string& dpath /*< path to dataset */,
                                     const string& epath /*< embeddings base path */);

};
