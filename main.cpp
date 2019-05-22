#include "face_engine.hpp"
#include <glib.h>

static gchar* dataset_path = NULL;
static gchar* predictor_path = NULL;
static gchar* fr_path = NULL;
static gboolean regenerate = FALSE;
static gchar* input_path = NULL;
static double threshold = 0.6;

GOptionEntry entries[] =
{
    {"dataset", 'd', 0, G_OPTION_ARG_FILENAME, &dataset_path, 
     "Designate path to pre-defined dataset", NULL},
    {"regenerate", 'r', 0, G_OPTION_ARG_NONE, &regenerate,
     "Regenerate embeddings from pre-defined dataset", NULL},
    {"predictor", 'p', 0, G_OPTION_ARG_FILENAME, &predictor_path,
     "Designate path to pre-trained shape predictor model file", NULL},
    {"face-model", 'f', 0, G_OPTION_ARG_FILENAME, &fr_path,
     "Designate path to pre-trained face model file", NULL}, 
    {"input", 'i', 0, G_OPTION_ARG_FILENAME, &input_path,
     "Input image file", NULL},
    {"threshold", 't', 0,  G_OPTION_ARG_DOUBLE, &threshold,
     "Threshold for recognition (0.0 ~ 1.0), smaller value forces a stricter judgement", NULL},
    {NULL}
};

void help(int argc, char** argv)
{
    printf("%s --help for options\n", argv[0]);
}


int main(int argc, char** argv)
{
    GError *error = NULL;
    GOptionContext *ctx = g_option_context_new("Face Recognition Demo");
    g_option_context_add_main_entries(ctx, entries, NULL);

    if (!g_option_context_parse(ctx, &argc, &argv, &error))
    {
        help(argc, argv);
        return -1;
    }

    g_option_context_free(ctx);

    if (regenerate && dataset_path == NULL)
    {
        printf("No dataset path designated to regenerate embeddings\n");
        help(argc, argv);
        return -1;
    }

    if (predictor_path == NULL)
    {
        printf("No pre-trained shape predictor model designated\n");
        help(argc, argv);
        return -1;
    }

    if (fr_path == NULL)
    {
        printf("No pre-trained face model designated\n");
        help(argc, argv);
        return -1;
    }

    FaceEngine engine;
    std::map<int, std::string> paths;
    paths[FaceEngine::DLIB_SHAPE_MODEL] = predictor_path;
    paths[FaceEngine::DLIB_FR_MODEL] = fr_path;
    if (!engine.InitializeModels(paths))
    {
        printf("Failed to load weight files\n");
        return -2;
    }

    image_window win;
    
    if (regenerate)
    {
        engine.BuildDataset(dataset_path);
    }

    if (input_path != NULL)
    {
        matrix<rgb_pixel> img;
        load_image(img, input_path);
        win.set_image(img);

        /* Want to do recognition, load embeddings */
        if (!engine.LoadEmbeddings())
        {
            printf("Failed to load embeddings\n");
            return -2;
        }

        std::vector<FaceEngine::Label> labels;
        if (!engine.Evaluate(img, labels, threshold))
        {
            printf("Failed to get labels\n");
            return -2;
        }

        for (auto label : labels)
        {
            win.add_overlay(label.bbox, rgb_pixel(255,0,0), label.name);
        }
    }

    printf("Press Enter to exit...");
    cin.get();
    return 0;
}
