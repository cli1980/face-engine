#include "face_engine.hpp"
#include <filesystem>
#include <vector>
#include <numeric> 

FaceEngine::FaceEngine()
{}

FaceEngine::~FaceEngine()
{}

bool FaceEngine::InitializeModels(const std::map<int, std::string> &wfiles)
{
    auto ishape = wfiles.find(DLIB_SHAPE_MODEL);
    if (wfiles.end() == ishape || !filesystem::exists(ishape->second))
    {
        cerr << " Missing weight file for shape model" << endl;
        return false;
    }

    auto ifr =  wfiles.find(DLIB_FR_MODEL);
    if (wfiles.end() == ifr || !filesystem::exists(ifr->second))
    {
        cerr << "Missing weight file for FR model" << endl;
        return false;
    }

    auto ifd = wfiles.find(DLIB_MMOD_MODEL);
    if (wfiles.end() == ifd || !filesystem::exists(ifd->second))
    {
        cout << "MMOD model not found, fall back to HOG detector" << endl;
        m_hog = get_frontal_face_detector();
        m_faceDetector = [this](const matrix<rgb_pixel>& img)
        {
            return m_hog(img);
        };
    }
    else
    {
        deserialize(ifd->second) >> m_mmodNet;
        m_faceDetector = [this](const matrix<rgb_pixel>& img)
        {
            auto &rects = m_mmodNet(img);
            std::vector<rectangle> res;
            for (auto &i : rects) res.push_back(i);
            return res;  
        };
    }

    deserialize(ishape->second) >> m_shapePred;
    deserialize(ifr->second) >> m_frNet;

    return true;
}

bool FaceEngine::BuildDataset(const string& dpath, const string& epath)
{
    if (!filesystem::exists(dpath))
    {
        cerr << "Dataset path " << dpath << " doesn't exist" << endl;
        return false;
    }

    if (filesystem::exists(epath))
    {
        /* erase current embeddings folder and its content*/
        filesystem::remove_all(epath);
    }
    filesystem::create_directory(epath);
    

    /* extract names in the dataset folder */
    std::vector<string> names;
    for (auto &entry : filesystem::directory_iterator(dpath))
    {
        if (entry.is_directory())
        {
            names.push_back(entry.path().filename().string());
        }
    }

    /* generate embeddings */
    for (auto &name : names)
    {
        _GenerateEmbeddings(name, dpath, epath);
    }

    return true;
}

bool FaceEngine::LoadEmbeddings(const string& path)
{
    filesystem::path epath(path);
    if (!filesystem::exists(epath))
    {
        cerr << "Path to embeddings not found here: " << epath.string() << endl;
        return false;
    }

    /* iterate the embeddings path for all given names */
    for (auto &entry : filesystem::directory_iterator(epath))
    {
        if (entry.is_directory())
        {
            auto name = entry.path().filename();
            cout << "Loading embeddings for " << name.string() << endl;
            std::vector<matrix<float, 0, 1>> embeddings;
            for (auto &i : filesystem::directory_iterator(entry.path()))
            {
                matrix<float, 0, 1> embedding;
                ifstream in(i.path().string(), ios_base::in);
                deserialize(embedding, in);
                embeddings.push_back(embedding);
            }
            if (embeddings.empty())
            {
                cout << "No embedding found for " << name.string() << endl;
                continue;
            }

            cout << embeddings.size() << " embeddings loaded for " << name.string() << endl;

            auto keypair = make_pair(name.string(), move(embeddings));
            auto [itr, inserted] = m_faceMap.insert(keypair);
            if (!inserted)
            {
                cout << name.string() << " already has its embeddings, skip duplicates" << endl;
            }

            for (auto embedding : m_faceMap[name.string()])
            {
                cout << trans(embedding) << endl;
            }
        }
    }

    return true;
}

bool FaceEngine::Evaluate(const matrix<rgb_pixel> &img, std::vector<Label>& labels, double threshold)
{
    cout << "Evaluating faces using threshold " << threshold << endl;
    
    std::vector<matrix<rgb_pixel>> faces;
    auto rectangles = m_faceDetector(img);
    for (auto &rect : rectangles)
    {
        auto shape = m_shapePred(img, rect);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        faces.push_back(move(face_chip));

        Label label = {rect, ""};
        labels.push_back(label);
    }

    auto count = faces.size();
    if (count == 0)
    {
        cout << "No face found" << endl;
        return false;
    }

    std::vector<matrix<float,0,1>> embeddings = m_frNet(faces);
    for (auto i = 0; i < count; i++)
    {
        /* evaluate a face on pre-defined dataset */

        std::map<std::string, std::vector<float>> result;
        matrix<float, 0, 1> embedding = embeddings[i];

        for (auto &entry : m_faceMap)
        {
            auto name = entry.first;
            auto candidates = entry.second;
            for (auto &candidate : candidates)
            {
                auto l = length(embedding - candidate);
                if (l < threshold)
                {
                    result[name].push_back(l);
                }
            }
        }

        string who = "unknown";
        auto max = 0;
        auto avg = 0.0;
        for (auto &entry : result)
        {
            auto &distances = entry.second;
            /* we only wanna find out the name with most hits */
            if (distances.size() < max) continue;

            auto tmp = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            if (distances.size() > max || tmp < avg)
            {
                max = distances.size();
                avg = tmp;
                who = entry.first;
                cout << "Distances against " << who << "[";
                for (auto d : distances) cout << d << " ";
                cout << "]" << endl;
            }
        }
        if (max != 0) 
        {
            cout << "Best match is " << who << ": " << max << " hits (avg=" << avg << ")" <<endl;
        }

        labels[i].name = who;
    }

    return true;
}

bool FaceEngine::_GenerateEmbeddings(const string& name, const string& dpath, const string &epath)
{
    /* extract faces from the dataset under the given name */
    std::vector<matrix<rgb_pixel>> faces;
    auto path = dpath / filesystem::path(name);
    for (auto &sample : filesystem::directory_iterator(path))
    {
        if (sample.is_regular_file() && sample.path().extension() == ".jpg")
        {
            cout << "Extracting face from sample " << sample.path().string() << endl;
            matrix<rgb_pixel> img;
            load_image(img, sample.path().string());

            auto rects = m_faceDetector(img);
            if (rects.size() != 1)
            {
                cout << "Sample not qualified, skipped" << endl;
                continue;
            }

            auto shape = m_shapePred(img, rects[0]);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
        }
    }

    cout << "Generating embeddings for " << name << " ..." << endl;
    std::vector<matrix<float,0,1>> embeddings = m_frNet(faces);

    auto kpath = epath / filesystem::path(name);
    if (!filesystem::exists(kpath))
    {
        filesystem::create_directory(kpath);
    }

    /* save all the embeddings under corresponding name */
    int count = 0;
    for (auto embedding : embeddings)
    {
        auto opath = kpath / to_string(count++);
        ofstream out(opath.string(), ios_base::out);
        serialize(embedding, out);
        cout << trans(embedding) << endl;
    }
    
    return true;    
}
