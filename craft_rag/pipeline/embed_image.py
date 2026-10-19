"""Optional CLIP-style image embedding interface; disabled by default."""
class ImageEmbedder:
    enabled = False
    def encode(self, image_paths):
        raise NotImplementedError("Image embedding is disabled; configure a CLIP-compatible model and licensed image records first.")
    @staticmethod
    def record(image_id, source_id, craft_cluster, why_theme, evidence_level, grade, image_path, caption="", **metadata):
        return {"chunk_id":image_id,"source_id":source_id,"craft_cluster":craft_cluster,"why_theme":why_theme,
                "secondary_theme":metadata.get("secondary_theme",""),"process_stage":metadata.get("process_stage",""),
                "evidence_level":evidence_level,"grade":grade,"questions_answered":metadata.get("questions_answered",[]),
                "text_chunk":caption,"species_material":metadata.get("species_material",""),
                "tool_treatment":metadata.get("tool_treatment",""),"link":metadata.get("link",""),
                "caution":metadata.get("caution","Image annotation pending."),"image_path":str(image_path),"modality":"image"}
