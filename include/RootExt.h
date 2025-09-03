/*! Common CERN ROOT extensions.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <map>
#include <memory>

#include <Compression.h>
#include <TClass.h>
#include <TFile.h>
#include <TH2.h>
#include <TLorentzVector.h>
#include <TMatrixD.h>
#include <TROOT.h>

#include "exception.h"

namespace root_ext {

    std::shared_ptr<TFile> inline CreateRootFile(const std::string &file_name,
                                                 ROOT::ECompressionAlgorithm compression = ROOT::kZLIB,
                                                 int compression_level = 9) {
        std::shared_ptr<TFile> file(
            TFile::Open(file_name.c_str(), "RECREATE", "", compression * 100 + compression_level));
        if (!file || file->IsZombie())
            throw analysis::exception("File '%1%' not created.") % file_name;
        return file;
    }

    std::shared_ptr<TFile> inline OpenRootFile(const std::string &file_name) {
        std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
        if (!file || file->IsZombie())
            throw analysis::exception("File '%1%' not opened.") % file_name;
        return file;
    }

    void WriteObject(const TObject &object, TDirectory *dir, const std::string &name = "") {
        if (!dir)
            throw analysis::exception("Can't write object to nullptr.");
        const std::string name_to_write = name.size() ? name : object.GetName();
        dir->WriteTObject(&object, name_to_write.c_str(), "Overwrite");
    }

    template <typename Object>
    void WriteObject(const Object &object) {
        TDirectory *dir = object.GetDirectory();
        WriteObject(object, dir);
    }

    template <typename Object>
    Object *ReadObject(TDirectory &file, const std::string &name) {
        if (!name.size())
            throw analysis::exception("Can't read nameless object.");
        TObject *root_object = file.Get(name.c_str());
        if (!root_object)
            throw analysis::exception("Object '%1%' not found in '%2%'.") % name % file.GetName();
        Object *object = dynamic_cast<Object *>(root_object);
        if (!object)
            throw analysis::exception("Wrong object type '%1%' for object '%2%' in '%3%'.") % typeid(Object).name() %
                name % file.GetName();
        return object;
    }

    template <typename Object>
    Object *TryReadObject(TDirectory &file, const std::string &name) {
        try {
            return ReadObject<Object>(file, name);
        } catch (analysis::exception &) {
        }
        return nullptr;
    }

    template <typename Object>
    Object *CloneObject(const Object &original_object, const std::string &new_name = "") {
        const std::string new_object_name = new_name.size() ? new_name : original_object.GetName();
        Object *new_object = dynamic_cast<Object *>(original_object.Clone(new_object_name.c_str()));
        if (!new_object)
            throw analysis::exception("Type error while cloning object '%1%'.") % original_object.GetName();
        return new_object;
    }

    template <typename Object>
    Object *CloneObject(const Object &original_object, const std::string &new_name, bool detach_from_file) {
        Object *new_object = CloneObject(original_object, new_name);
        if (detach_from_file)
            new_object->SetDirectory(nullptr);
        return new_object;
    }

    template <typename Object>
    Object *ReadCloneObject(TDirectory &file,
                            const std::string &original_name,
                            const std::string &new_name = "",
                            bool detach_from_file = false) {
        Object *original_object = ReadObject<Object>(file, original_name);
        return CloneObject(*original_object, new_name, detach_from_file);
    }

    TDirectory *GetDirectory(TDirectory &root_dir, const std::string &name, bool create_if_needed = true) {
        if (!name.size() || (name.size() == 1 && name.at(0) == '/'))
            return &root_dir;
        TDirectory *dir = root_dir.GetDirectory(name.c_str());
        if (!dir && create_if_needed) {
            const size_t pos = name.find("/");
            if (pos == std::string::npos || pos == name.size() - 1) {
                root_dir.mkdir(name.c_str());
                dir = root_dir.GetDirectory(name.c_str());
            } else {
                const std::string first_dir_name = name.substr(0, pos), sub_dirs_path = name.substr(pos + 1);
                TDirectory *first_dir = GetDirectory(root_dir, first_dir_name, true);
                dir = GetDirectory(*first_dir, sub_dirs_path, true);
            }
        }

        if (!dir)
            throw analysis::exception("Unable to get directory '%1%' from the root directory '%2%'.") % name %
                root_dir.GetName();
        return dir;
    }

    enum class ClassInheritance { TH1, TTree, TDirectory };

    ClassInheritance FindClassInheritance(const std::string &class_name) {
        TClass *cl = gROOT->GetClass(class_name.c_str());
        if (!cl)
            throw analysis::exception("Unable to get TClass for class named '%1%'.") % class_name;

        ClassInheritance inheritance;
        if (cl->InheritsFrom("TH1"))
            inheritance = ClassInheritance::TH1;
        else if (cl->InheritsFrom("TTree"))
            inheritance = ClassInheritance::TTree;
        else if (cl->InheritsFrom("TDirectory"))
            inheritance = ClassInheritance::TDirectory;
        else
            throw analysis::exception("Unknown class inheritance for class named '%1%'.") % class_name;

        return inheritance;
    }

    struct WarningSuppressor {
        const Int_t old_ignore_level;
        WarningSuppressor(Int_t ignore_level) : old_ignore_level(gErrorIgnoreLevel) {
            gErrorIgnoreLevel = ignore_level;
        }
        ~WarningSuppressor() { gErrorIgnoreLevel = old_ignore_level; }
    };

}  // namespace root_ext