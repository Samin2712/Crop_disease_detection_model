import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { requestsApi } from "@/api/requests";

export default function CreateRequestPage() {
  const navigate = useNavigate();

  const [category, setCategory] = useState("");
  const [description, setDescription] = useState("");
  const [state, setState] = useState("");
  const [district, setDistrict] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    if (!category.trim() || !description.trim()) return;

    setSaving(true);
    try {
      const created = await requestsApi.create({
        category: category.trim(),
        description: description.trim(),
        state: state.trim() || undefined,
        district: district.trim() || undefined,
        image,
      });

      navigate(`/requests/${created.id}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle>Create a new request</CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">Category</div>
            <Input value={category} onChange={(e) => setCategory(e.target.value)} placeholder="e.g. Leaf disease" />
          </div>

          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">Description</div>
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Explain the issue..." />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">State</div>
              <Input value={state} onChange={(e) => setState(e.target.value)} placeholder="Optional" />
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">District</div>
              <Input value={district} onChange={(e) => setDistrict(e.target.value)} placeholder="Optional" />
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">Photo (optional)</div>
            <Input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} />
          </div>

          <Button disabled={saving || !category.trim() || !description.trim()} onClick={submit}>
            {saving ? "Submitting..." : "Submit request"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
